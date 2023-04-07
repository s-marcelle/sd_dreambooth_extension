"""
extract factors the build is dependent on:
[X] compute capability
    [ ] TODO: Q - What if we have multiple GPUs of different makes?
- CUDA version
- Software:
    - CPU-only: only CPU quantization functions (no optimizer, no matrix multipl)
    - CuBLAS-LT: full-build 8-bit optimizer
    - no CuBLAS-LT: no 8-bit matrix multiplication (`nomatmul`)

evaluation:
    - if paths faulty, return meaningful error
    - else:
        - determine CUDA version
        - determine capabilities
        - based on that set the default path
"""

import ctypes
import os

from .paths import determine_cuda_runtime_lib_path

CUDA_RUNTIME_LIB: str = "libcudart.so"

class CUDASetup:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def generate_instructions(self):
        if self.cuda is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1): Your paths are probably not up-to-date. You can update them via: sudo ldconfig.')
            self.add_log_entry('CUDA SETUP: Solution 2): If you do not have sudo rights, you can do the following:')
            self.add_log_entry('CUDA SETUP: Solution 2a): Find the cuda library via: find / -name libcuda.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 2b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_2a')
            self.add_log_entry('CUDA SETUP: Solution 2c): For a permanent solution add the export from 2b into your .bashrc file, located at ~/.bashrc')
            return

        if self.cudart_path is None:
            self.add_log_entry('CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.')
            self.add_log_entry('CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable')
            self.add_log_entry('CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null')
            self.add_log_entry('CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a')
            self.add_log_entry('CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc')
            self.add_log_entry('CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.')
            self.add_log_entry('CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh')
            self.add_log_entry('CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.')
            self.add_log_entry('CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local')
            return

        make_cmd = f'CUDA_VERSION={self.cuda_version_string}'
        if len(self.cuda_version_string) < 3:
            make_cmd += ' make cuda92'
        elif self.cuda_version_string == '110':
            make_cmd += ' make cuda110'
        elif self.cuda_version_string[:2] == '11' and int(self.cuda_version_string[2]) > 0:
            make_cmd += ' make cuda11x'
        elif self.cuda_version_string == '100':
            self.add_log_entry('CUDA SETUP: CUDA 10.0 not supported. Please use a different CUDA version.')
            self.add_log_entry('CUDA SETUP: Before you try again running bitsandbytes, make sure old CUDA 10.0 versions are uninstalled and removed from $LD_LIBRARY_PATH variables.')
            return


        has_cublaslt = is_cublasLt_compatible(self.cc)
        if not has_cublaslt:
            make_cmd += '_nomatmul'

        self.add_log_entry('CUDA SETUP: Something unexpected happened. Please compile from source:')
        self.add_log_entry('git clone git@github.com:TimDettmers/bitsandbytes.git')
        self.add_log_entry('cd bitsandbytes')
        self.add_log_entry(make_cmd)
        self.add_log_entry('python setup.py install')

    def initialize(self):
        if not getattr(self, 'initialized', False):
            self.has_printed = False
            self.lib = None
            self.initialized = False

    def run_cuda_setup(self):
        self.initialized = True
        self.cuda_setup_log = []

        binary_name, cudart_path, cuda, cc, cuda_version_string = evaluate_cuda_setup()
        self.cudart_path = cudart_path
        self.cuda = cuda
        self.cc = cc
        self.cuda_version_string = cuda_version_string

        package_dir = Path(__file__).parent.parent
        binary_path = package_dir / binary_name

        try:
            if not binary_path.exists():
                self.add_log_entry(f"CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?")
                legacy_binary_name = "libbitsandbytes_cpu.so"
                self.add_log_entry(f"CUDA SETUP: Defaulting to {legacy_binary_name}...")
                binary_path = package_dir / legacy_binary_name
                if not binary_path.exists() or torch.cuda.is_available():
                    self.add_log_entry('')
                    self.add_log_entry('='*48 + 'ERROR' + '='*37)
                    self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                    self.add_log_entry('1. CUDA driver not installed')
                    self.add_log_entry('2. CUDA not installed')
                    self.add_log_entry('3. You have multiple conflicting CUDA libraries')
                    self.add_log_entry('4. Required library not pre-compiled for this bitsandbytes release!')
                    self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=113`.')
                    self.add_log_entry('CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.')
                    self.add_log_entry('='*80)
                    self.add_log_entry('')
                    self.generate_instructions()
                    self.print_log_stack()
                    raise Exception('CUDA SETUP: Setup Failed!')
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
            else:
                self.add_log_entry(f"CUDA SETUP: Loading binary {binary_path}...")
                self.lib = ct.cdll.LoadLibrary(str(binary_path))
        except Exception as ex:
            self.add_log_entry(str(ex))
            self.print_log_stack()

    def add_log_entry(self, msg, is_warning=False):
        self.cuda_setup_log.append((msg, is_warning))

    def print_log_stack(self):
        for msg, is_warning in self.cuda_setup_log:
            if is_warning:
                warn(msg)
            else:
                print(msg)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance


def check_cuda_result(cuda, result_val):
    # 3. Check for CUDA errors
    if result_val != 0:
        error_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result_val, ctypes.byref(error_str))
        print(f"CUDA exception! Error code: {error_str.value.decode()}")


def get_cuda_version(cuda, cudart_path):
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
    try:
        cudart = ctypes.CDLL(cudart_path)
    except OSError:
        # TODO: shouldn't we error or at least warn here?
        print(f'ERROR: libcudart.so could not be read from path: {cudart_path}!')
        return None

    version = ctypes.c_int()
    check_cuda_result(cuda, cudart.cudaRuntimeGetVersion(ctypes.byref(version)))
    version = int(version.value)
    major = version // 1000
    minor = (version - (major * 1000)) // 10

    if major < 11:
        print(
            'CUDA SETUP: CUDA version lower than 11 are currenlty not supported for LLM.int8(). You will be only to use 8-bit optimizers and quantization routines!!')

    return f'{major}{minor}'


def get_cuda_lib_handle():
    # 1. find libcuda.so library (GPU driver) (/usr/lib)
    try:
        cuda = ctypes.CDLL("libcuda.so")
    except OSError:
        # TODO: shouldn't we error or at least warn here?
        print(
            'CUDA SETUP: WARNING! libcuda.so not found! Do you have a CUDA driver installed? If you are on a cluster, make sure you are on a CUDA machine!')
        return None
    check_cuda_result(cuda, cuda.cuInit(0))

    return cuda


def get_compute_capabilities(cuda):
    """
    1. find libcuda.so library (GPU driver) (/usr/lib)
       init_device -> init variables -> call function by reference
    2. call extern C function to determine CC
       (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html)
    3. Check for CUDA errors
       https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    # bits taken from https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
    """

    nGpus = ctypes.c_int()
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    device = ctypes.c_int()

    check_cuda_result(cuda, cuda.cuDeviceGetCount(ctypes.byref(nGpus)))
    ccs = []
    for i in range(nGpus.value):
        check_cuda_result(cuda, cuda.cuDeviceGet(ctypes.byref(device), i))
        ref_major = ctypes.byref(cc_major)
        ref_minor = ctypes.byref(cc_minor)
        # 2. call extern C function to determine CC
        check_cuda_result(
            cuda, cuda.cuDeviceComputeCapability(ref_major, ref_minor, device)
        )
        ccs.append(f"{cc_major.value}.{cc_minor.value}")

    return ccs


# def get_compute_capability()-> Union[List[str, ...], None]: # FIXME: error
def get_compute_capability(cuda):
    """
    Extracts the highest compute capbility from all available GPUs, as compute
    capabilities are downwards compatible. If no GPUs are detected, it returns
    None.
    """
    ccs = get_compute_capabilities(cuda)
    if ccs is not None:
        # TODO: handle different compute capabilities; for now, take the max
        return ccs[-1]
    return None


def evaluate_cuda_setup():
    if os.name == "nt":
        return "libbitsandbytes_cudaall.dll"  # $$$

    binary_name = "libbitsandbytes_cpu.so"
    # if not torch.cuda.is_available():
    # print('No GPU detected. Loading CPU library...')
    # return binary_name

    cudart_path = determine_cuda_runtime_lib_path()
    if cudart_path is None:
        print(
            "WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package (anaconda)!"
        )
        return binary_name

    print(f"CUDA SETUP: CUDA runtime path found: {cudart_path}")
    cuda = get_cuda_lib_handle()
    cc = get_compute_capability(cuda)
    print(f"CUDA SETUP: Highest compute capability among GPUs detected: {cc}")
    cuda_version_string = get_cuda_version(cuda, cudart_path)

    if cc == '':
        print(
            "WARNING: No GPU detected! Check your CUDA paths. Processing to load CPU-only library..."
        )
        return binary_name

    # 7.5 is the minimum CC vor cublaslt
    has_cublaslt = cc in ["7.5", "8.0", "8.6"]

    # TODO:
    # (1) CUDA missing cases (no CUDA installed by CUDA driver (nvidia-smi accessible)
    # (2) Multiple CUDA versions installed

    # we use ls -l instead of nvcc to determine the cuda version
    # since most installations will have the libcudart.so installed, but not the compiler
    print(f'CUDA SETUP: Detected CUDA version {cuda_version_string}')

    def get_binary_name():
        """if not has_cublaslt (CC < 7.5), then we have to choose  _nocublaslt.so"""
        bin_base_name = "libbitsandbytes_cuda"
        if has_cublaslt:
            return f"{bin_base_name}{cuda_version_string}.so"
        else:
            return f"{bin_base_name}{cuda_version_string}_nocublaslt.so"

    binary_name = get_binary_name()

    return binary_name
