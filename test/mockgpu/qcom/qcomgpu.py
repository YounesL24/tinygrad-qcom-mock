from test.mockgpu.gpu import VirtGPU

class CommandObject:
    def __init__(self, gpuaddr, size):
        self.gpuaddr = gpuaddr
        self.size = size

class QCOMGPU(VirtGPU):
    def __init__(self, gpu_id):
        super().__init__(gpu_id)
        self.arch = "a630"
        self.devinfo = {
            "device_id": 0xC0DEFACE,
            "chip_id": 0x06030000,  # a630 emulator (6, 3, 0, 0)
            "mmu_enabled": True,
            "gmem_gpubaseaddr": 0x00100000,
            "gmem_sizebytes": 0x00100000
        }
        self.mapped_ranges = []
        self.next_cmd_id = 1
        self.queue = {}
    def map_range(self, vaddr, size):
        self.mapped_ranges.append((vaddr, size))
    def unmap_range(self, vaddr, size):
        self.mapped_ranges.remove((vaddr, size))
    def submit_command(self, cmd_obj):
        cmd_id = self.next_cmd_id
        self.queue[self.next_cmd_id] = cmd_obj
        self.next_cmd_id += 1
        return cmd_id