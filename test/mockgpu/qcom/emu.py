import ctypes
import functools
import itertools
import math
from typing import Callable

from tinygrad.codegen import to_program
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import DEBUG, Context, ceildiv, colored
from tinygrad.uop.ops import UOp, Ops
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

def _c(val, dtype=dtypes.uint32): return UOp.const(val, dtype)

class Inst:
    ...

class _Ctx:
    ...

def _compile_inst(inst, ctx):
    ...

_canonical_runner_cache: dict[Inst, (UOp, object)] = {}  # [(inst, (prg, runtime))]

@functools.cache
def _get_runner(inst):
    if _canonical_runner_cache.get(inst, None) is not None: return _canonical_runner_cache[inst]
    ctx = _Ctx(...) #TODO: tbd
    sink = _compile_inst(inst, ctx)
    with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
        prg = to_program(sink, Device['CPU'].renderer)
        runtime = get_runtime('CPU', prg)
    _canonical_runner_cache[inst] = (prg, runtime)
    return prg, runtime
def _decode_inst(inst_code):
    #TODO: create instruction class and decode based on bits
    ...

def _decode_at(pc, shader):
    inst_low = shader[pc]
    inst_high = shader[pc + 1]
    inst_code = (inst_high << 32) | inst_low
    inst = _decode_inst(inst_code)
    try: return _get_runner(inst), inst
    except Exception as e:
        try: inst_str = repr(inst)
        except Exception: inst_str = f"<{type(inst).__name__}>"
        raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e
    

class WaveState:
    #TODO: regs may be better placed elsewhere (_Ctx, and as buffers passed in in run_asm)
    __slots__ = {"wave_size", "wave_idx", "regs", "_axis_id", "pc"}
    #cache of some sort?

    def _decode_reg(self, reg_id):
        return reg_id >> 2, reg_id & 3 

    def get_reg(self, reg_type, reg_num, reg_dim):
        return self.regs[(reg_type, reg_num, reg_dim)]

    def set_reg(self, reg_type, reg_num, reg_dim, value):
        self.regs[(reg_type, reg_num, reg_dim)] = value

    def range(self, n = None):
        if n is None: n = self.wave_size
        self._axis_id += 1
        return UOp.range(n, self._axis_id, dtype=dtypes.int)

    def __init__(
        self, wave_size, wave_idx,
        wgidconstid, wgsizeconstid, localidregid,
        wgid, wgsize, local_dims
    ):
        self.wave_size = wave_size
        self.wave_idx = wave_idx
        self._axis_id = 0
        self.regs = {}
        self.pc = 0

        wgid_reg, _ = self._decode_reg(wgidconstid)
        for i in range(3):
            self.regs[("c", wgid_reg, i)] = _c(wgid[i])
        wgsize_reg, _ = self._decode_reg(wgsizeconstid)
        for i in range(3):
            self.regs[("c", wgsize_reg, i)] = _c(wgsize[i])

        localid_reg, _ = self._decode_reg(localidregid)
        lane = self.range(wave_size) + _c(wave_idx * wave_size)
        local_x = lane % _c(local_dims[0])
        local_y = (lane // _c(local_dims[0])) % _c(local_dims[1])
        local_z = lane // (_c(local_dims[0]) * _c(local_dims[1]))
        localid = (local_x, local_y, local_z)
        for i in range(3):
            self.regs[("r", localid_reg, i)] = localid[i]
        #wgoffset and linearlocalid not exposed

def _init_wave(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    wave_start, total_threads, local_dims, pvt_memsize_peritem
):
    return

def run_asm(
    shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
    global_dims, group_dims, local_dims, kerneldim, wave_size,
    shared_memsize, pvt_memsize_peritem, pvt_mem_base,
    wgidconstid, wgsizeconstid, wgoffsetconstid, localidregid, linearlocalidregid
):

    program: dict[int, tuple[Callable, list[int], bool, Inst]] = {} #TODO: clarify types

    total_threads = math.prod(local_dims)


    vmem_buf = Buffer('CPU', 1<<40, dtypes.uint8, options=BufferSpec(external_ptr=0)).ensure_allocated()
    shared_buf = Buffer('CPU', max(shared_memsize, 1), dtypes.uint8).ensure_allocated()

    n_waves = ceildiv(total_threads, wave_size)
    #TODO remember to subtract pvt_mem_base from private mem accesses
    private_buf = Buffer('CPU', pvt_memsize_peritem * wave_size * n_waves, dtypes.uint8).ensure_allocated() if pvt_memsize_peritem > 0 else None

    def _ensure_compiled(pc):
        if pc not in program:
            (prg, runtime), inst = _decode_at(pc, shader)
            is_barrier = ... #TODO: determine barrier instructions
            program[pc] = (runtime.fxn, prg.arg.globals, is_barrier, inst)
            if DEBUG >= 3:
                prev_len = len(_canonical_runner_cache)
                msg = f"[emu] PC={pc}: {inst!r}"
                print(colored(msg, 'green') if len(_canonical_runner_cache) > prev_len else msg)
        return program[pc]

    def _run_workgroup(group_id):
        waves = []
        for wave_start in range(0, total_threads, wave_size):
            st = _init_wave(
                shader, shader_constants, shader_uav, shader_tex, shader_tex_constants,
                wave_start, total_threads, local_dims, pvt_memsize_peritem
            )
            waves.append(
                (st,
                    [
                        ctypes.c_uint64(vmem_buf._buf),
                        ctypes.c_uint64(shared_buf._buf),
                        ctypes.c_uint64(pvt_mem_base if private_buf else 0)
                    ]
                )
            )
        done = [False] * len(waves)
        for _ in range(10_000_000):
            if all(done): return
            for wi, (st, bufs) in enumerate(waves):
                if done[wi]: continue
                for _ in range(1_000_000):
                    pc = st.pc
                    fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)    
                    #TODO done detection? probably using end opcode, or instruction list size
                    if DEBUG >= 5: print(f"  exec gid=({group_id[0]},{group_id[1]},{group_id[2]}) w={wi} PC={pc}: {inst!r}", flush=True)
                    fxn(*[bufs[g] for g in globals_list])
                    if is_barrier: break
                else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
        raise RuntimeError("exceeded 10M total scheduling rounds")
    (group_x, group_y, group_z) = group_dims

    for gidz, gidy, gidx in itertools.product(range(group_z), range(group_y), range(group_x)):
        _run_workgroup((gidx, gidy, gidz))



