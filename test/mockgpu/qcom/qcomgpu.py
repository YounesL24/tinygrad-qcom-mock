from test.mockgpu.gpu import VirtGPU
from tinygrad.runtime.autogen import kgsl
import ctypes

class QCOMCommandObject:
    def __init__(self, numcmds, cmdlist, cmdsize):
        self.numcmds = numcmds
        self.cmdlist = cmdlist
        self.cmdsize = cmdsize

class QCOMGPU(VirtGPU):
    def __init__(self, gpu_id):
        super().__init__(gpu_id)
        self.arch = "a630"
        self.devinfo = {
            "device_id": 0xC0DEFACE,
            "chip_id": 0x06030000,  # a630 emulator (6, 3, 0, 0)
            "mmu_enabled": 1,
            "gmem_gpubaseaddr": 0x00100000,
            "gpu_id": gpu_id,
            "gmem_sizebytes": 0x00100000
        }
        self.mapped_ranges = set()
        self.queue = []
    def map_range(self, vaddr, size):
        self.mapped_ranges.add((vaddr, size))
    def unmap_range(self, vaddr, size):
        self.mapped_ranges.remove((vaddr, size))
    def submit_command(self, numcmds, cmdlist, cmdsize):
        cmd_struct = kgsl.struct_kgsl_command_object.from_address(cmdlist)
        if not cmd_struct.flags & kgsl.KGSL_CMDLIST_IB: raise NotImplementedError("Only KGSL_CMDLIST_IB is supported")
        print(f"received {numcmds} commands, at gpu addr {cmd_struct.gpuaddr:#x}, each of size {cmd_struct.size:#x}")
        # for i in range(numcmds):
        #     cmd_obj = kgsl.struct_kgsl_command_object.from_address(cmdlist.gpuaddr + i * kgsl.struct_kgsl_ibdesc.SIZE)
        #     print(f"  command {i}: gpuaddr={cmd_obj.gpuaddr:#x}")
        self.queue.append(QCOMCommandObject(numcmds, cmdlist, cmdsize))
        return len(self.queue) - 1