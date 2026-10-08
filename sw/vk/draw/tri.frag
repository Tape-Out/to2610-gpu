#version 460
layout(location = 0) flat in uint col;
layout(location = 0) out uint frag;

void main() { frag = col; }
