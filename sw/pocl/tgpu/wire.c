#include "wire.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

struct wire
{
  int fd;
  int tcp;
  uint32_t hz;
};

static int
dial (const char *hostport)
{
  const char *colon = strrchr (hostport, ':');
  char host[256];
  if (!colon || (size_t)(colon - hostport) >= sizeof host)
    return -1;
  memcpy (host, hostport, colon - hostport);
  host[colon - hostport] = 0;

  struct addrinfo hints = { .ai_socktype = SOCK_STREAM }, *res;
  if (getaddrinfo (host, colon + 1, &hints, &res))
    return -1;
  int fd = -1;
  for (struct addrinfo *ai = res; ai && fd < 0; ai = ai->ai_next)
    {
      fd = socket (ai->ai_family, ai->ai_socktype, ai->ai_protocol);
      if (fd >= 0 && connect (fd, ai->ai_addr, ai->ai_addrlen))
        {
          close (fd);
          fd = -1;
        }
    }
  freeaddrinfo (res);
  /* 每次传输都要等应答才发下一次，Nagle 会让每一来回多等一个 ACK 的延迟 */
  int one = 1;
  if (fd >= 0)
    setsockopt (fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  return fd;
}

static int
spidev (const char *spec, uint32_t *hz)
{
  const char *at = strchr (spec, '@');
  size_t n = at ? (size_t)(at - spec) : strlen (spec);
  char path[256];
  if (n >= sizeof path)
    return -1;
  memcpy (path, spec, n);
  path[n] = 0;
  *hz = at ? strtoul (at + 1, NULL, 0) : 2000000;

  int fd = open (path, O_RDWR);
  uint8_t mode = SPI_MODE_0;
  if (fd >= 0 && ioctl (fd, SPI_IOC_WR_MODE, &mode) < 0)
    {
      close (fd);
      fd = -1;
    }
  return fd;
}

wire *
wire_open (const char *spec)
{
  wire *l = calloc (1, sizeof *l);
  if (!l)
    return NULL;
  if (!strncmp (spec, "tcp:", 4))
    {
      l->tcp = 1;
      l->fd = dial (spec + 4);
    }
  else if (!strncmp (spec, "spidev:", 7))
    l->fd = spidev (spec + 7, &l->hz);
  else
    l->fd = -1;
  if (l->fd < 0)
    {
      free (l);
      return NULL;
    }
  return l;
}

void
wire_close (wire *l)
{
  if (l)
    close (l->fd);
  free (l);
}

static int
whole (int fd, void *buf, size_t n, int out)
{
  for (uint8_t *p = buf; n;)
    {
      ssize_t k = out ? write (fd, p, n) : read (fd, p, n);
      if (k < 0 && errno == EINTR)
        continue;
      if (k <= 0)
        return -1;
      p += k;
      n -= k;
    }
  return 0;
}

int
wire_xfer (wire *l, const uint8_t *tx, uint8_t *rx, size_t n)
{
  if (l->tcp)
    {
      uint8_t h[4] = { n >> 24, n >> 16, n >> 8, n };
      return whole (l->fd, h, 4, 1) || whole (l->fd, (void *)tx, n, 1)
                     || whole (l->fd, rx, n, 0)
                 ? -1
                 : 0;
    }
  struct spi_ioc_transfer t = { .tx_buf = (uintptr_t)tx,
                                .rx_buf = (uintptr_t)rx,
                                .len = n,
                                .speed_hz = l->hz,
                                .bits_per_word = 8 };
  return ioctl (l->fd, SPI_IOC_MESSAGE (1), &t) < 0 ? -1 : 0;
}

static void
be32 (uint8_t *p, uint32_t v)
{
  p[0] = v >> 24;
  p[1] = v >> 16;
  p[2] = v >> 8;
  p[3] = v;
}

static uint32_t
get32 (const uint8_t *p)
{
  return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3];
}

/* 一帧最多 256 个字，落在 spidev 默认 4 KiB 的缓冲里 */
enum { CHUNK = 256 };

int
spis_wr (wire *l, uint32_t addr, const uint32_t *w, size_t n)
{
  uint8_t tx[5 + 4 * CHUNK], rx[sizeof tx];
  while (n)
    {
      size_t k = n < CHUNK ? n : CHUNK;
      tx[0] = 0x02;
      be32 (tx + 1, addr);
      for (size_t i = 0; i < k; ++i)
        be32 (tx + 5 + 4 * i, w[i]);
      if (wire_xfer (l, tx, rx, 5 + 4 * k))
        return -1;
      addr += 4 * k;
      w += k;
      n -= k;
    }
  return 0;
}

int
spis_rd (wire *l, uint32_t addr, uint32_t *w, size_t n)
{
  uint8_t tx[7 + 4 * CHUNK], rx[sizeof tx];
  while (n)
    {
      size_t k = n < CHUNK ? n : CHUNK, len, at;
      memset (tx, 0, sizeof tx);
      be32 (tx + 1, addr);
      if (k == 1)
        {
          tx[0] = 0x03;
          len = 10;
          at = 6;
        }
      else
        {
          tx[0] = 0x0B;
          tx[5] = k - 1;
          len = 7 + 4 * k;
          at = 7;
        }
      if (wire_xfer (l, tx, rx, len))
        return -1;
      for (size_t i = 0; i < k; ++i)
        w[i] = get32 (rx + at + 4 * i);
      addr += 4 * k;
      w += k;
      n -= k;
    }
  return 0;
}
