"""给 sv2v 出来的 gpu.v 补上上游例化时漏接的输入。

上游例化程序存储的控制器时只接了读的那一半，写的那一半（consumer_write_valid 等四个输入）悬空，
而控制器里照样在看它们：参数 WRITE_ENABLE 只声明了，没用在任何条件里。上游自己的仿真里悬空是 X，
`if` 把 X 当假，碰巧不出事；综合或展平之后悬空的输入是工具挑的常数，仿真里则是一路传下去的 X，
控制器的状态当场变 X，一条指令也取不出来。这里把它们明写成 0。

用法：tieoff.py <gpu.v>，就地改。锚点找不到或不止一处就报错，不悄悄放过。
"""
import pathlib
import sys

ANCHOR = ") program_memory_controller(\n"
TIES = (
    ("consumer_write_valid", "NUM_FETCHERS"),
    ("consumer_write_address", "NUM_FETCHERS * PROGRAM_MEM_ADDR_BITS"),
    ("consumer_write_data", "NUM_FETCHERS * PROGRAM_MEM_DATA_BITS"),
    ("mem_write_ready", "PROGRAM_MEM_NUM_CHANNELS"),
)

p = pathlib.Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
if s.count(ANCHOR) != 1:
    sys.exit(f"{p}：program_memory_controller 的例化出现了 {s.count(ANCHOR)} 处，补不了")
for name, _ in TIES:
    if f".{name}(" in s[s.index(ANCHOR):s.index(");", s.index(ANCHOR))]:
        sys.exit(f"{p}：上游已经接了 {name}，这一步该撤了")
tie = "".join(f"\t\t.{name}({{({width}) {{1'b0}}}}),\n" for name, width in TIES)
p.write_text(s.replace(ANCHOR, ANCHOR + tie), encoding="utf-8")
