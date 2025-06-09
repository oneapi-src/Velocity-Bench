# This file is part of Leela Chess Zero.
# Copyright (C) 2018-2022 The LCZero Authors
#
# Leela Chess is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Leela Chess is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.


#    This file is part of Leela Chess Zero.
#    Modifications Copyright (C) 2023 Intel Corporation
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#    SPDX license identifier to be GNU General Public License v3.0 or later


project('lc0', 'cpp',
        #default_options : ['cpp_std=c++17', 'b_ndebug=if-release', 'warning_level=3', 'b_lto=true', 'b_vscrt=mt', ],
        default_options : ['cpp_std=c++20', 'b_ndebug=false'],
        meson_version: '>=0.52')


cc = meson.get_compiler('cpp')

if not cc.has_header('optional') or not cc.has_header('string_view')
    error('Lc0 requires a compiler supporting C++17, for example g++ v8.0, ' +
          'clang v5.0 or later (with C++17 stdlib) and Visual Studio 2017 or ' +
          'later.')
endif

if not cc.has_header('charconv')
    warning('Your compiler or library does not have full C++17 support. ' +
            'See the README for compilers that are known to be working. ' +
            'This will become an error in the future.')
endif

if cc.get_id() == 'clang'
  # Thread safety annotation
 # add_project_arguments('-Wthread-safety', language : 'cpp')
endif

#if cc.get_id() == 'clang' or cc.get_id() == 'gcc'
#  if get_option('buildtype') == 'release'
#    add_project_arguments(cc.get_supported_arguments(['-march=native']), language : 'cpp')
#  endif
#endif



# Files to compile.
deps = []
files = []
includes = []
has_backends = false


#add_project_arguments('-O2', language : 'cpp')
#add_project_arguments('-fsycl', language : 'cpp')
#add_project_arguments('-ffast-math', language : 'cpp')
#add_project_arguments('-fsycl-unnamed-lambda', language : 'cpp') 


#if get_option('nvidia')
#add_project_arguments('-fsycl-targets=nvptx64-nvidia-cuda', language : 'cpp') 
#add_project_arguments('-Xsycl-target-backend --cuda-gpu-arch=sm_61', language : 'cpp') 
#add_project_arguments('-DUSE_CUBLAS', language : 'cpp') 
#else
#add_project_arguments('-fsycl-targets=spir64_gen -Xs -device 0x0bd5 -revision_id 3' ', language : 'cpp') 
#endif

# Common includes
includes +=  include_directories('../infrastructure')

# Third party files.
includes += include_directories('third_party', is_system: true)

# Compiling protobufs.
compile_proto = find_program('scripts/compile_proto.py')
gen = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=@CURRENT_SOURCE_DIR@/libs/lczero-common',
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

# Handle submodules.
git = find_program('git', required: false)
if run_command('scripts/checkdir.py', 'libs/lczero-common/proto').returncode() != 0
  if git.found()
    if run_command(git, 'status').returncode() == 0
      message('updating git submodule libs/lczero-common')
      run_command(git, 'submodule', 'update', '--init', '--recursive')
    else
      message('cloning lczero-common.git into libs/lczero-common')
      run_command(git, 'clone', '--depth=1',
                  'https://github.com/LeelaChessZero/lczero-common.git',
                  'libs/lczero-common/')
    endif
  else
    error('Please install git to automatically fetch submodules or download the archives manually from GitHub.')
  endif
endif

pb_files = [
  'src/utils/protomessage.cc',
  gen.process('libs/lczero-common/proto/net.proto',
    preserve_path_from : meson.current_source_dir() + '/libs/lczero-common/')
]
files += pb_files

# Extract git short revision.
short_rev = 'unknown'
if git.found()
  r = run_command(git, 'rev-parse', '--short', 'HEAD')
  if r.returncode() == 0
    # Now let's check if the working directory is clean.
    if run_command(git, 'diff-index', '--quiet', 'HEAD').returncode() == 0
      short_rev = r.stdout().strip()
      if run_command(git, 'describe', '--exact-match', '--tags').returncode() == 0
        short_rev = ''
      endif
    else
      short_rev = 'dirty'
      warning('Cannot extract valid git short revision from dirty working directory.')
    endif
  else
    warning('Failed to parse short revision. Use git clone instead of downloading the archive from GitHub.')
  endif
endif

# Construct build identifier.
build_identifier = ''
if short_rev != ''
  build_identifier = 'git.' + short_rev
  message('Using build identifier "' + build_identifier + '".')
endif

conf_data = configuration_data()
conf_data.set_quoted('BUILD_IDENTIFIER', build_identifier)
configure_file(output: 'build_id.h', configuration: conf_data)

# Some malloc libraries require to be linked first.
if get_option('malloc') == 'mimalloc' and cc.get_id() == 'msvc'
  if get_option('b_vscrt') != 'md' and get_option('b_vscrt') != 'mdd'
    error('You need -Db_vscrt=md (or mdd)')
  endif
  add_project_link_arguments('/INCLUDE:mi_version', language : 'cpp')
  deps += cc.find_library('mimalloc-override', dirs: get_option('mimalloc_libdir'), required: true)
elif get_option('malloc') != ''
  deps += cc.find_library(get_option('malloc'), required: true)
endif

# ONNX protobufs.
gen_proto_src = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=@CURRENT_SOURCE_DIR@/src',
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

files += gen_proto_src.process('src/neural/onnx/onnx.proto',
  preserve_path_from : meson.current_source_dir() + '/src/')

#############################################################################
## Main files
#############################################################################
files += [
  'src/benchmark/backendbench.cc',
  'src/benchmark/benchmark.cc',
  'src/chess/bitboard.cc',
  'src/chess/board.cc',
  'src/chess/position.cc',
  'src/chess/uciloop.cc',
  'src/engine.cc',
  'src/lc0ctl/describenet.cc',
  'src/lc0ctl/leela2onnx.cc',
  'src/lc0ctl/onnx2leela.cc',  
  'src/mcts/node.cc',
  'src/mcts/params.cc',
  'src/mcts/search.cc',
  'src/mcts/stoppers/alphazero.cc',
  'src/mcts/stoppers/common.cc',
  'src/mcts/stoppers/factory.cc',
  'src/mcts/stoppers/legacy.cc',
  'src/mcts/stoppers/simple.cc',
  'src/mcts/stoppers/smooth.cc',
  'src/mcts/stoppers/stoppers.cc',
  'src/mcts/stoppers/timemgr.cc',
  'src/neural/cache.cc',
  'src/neural/decoder.cc',
  'src/neural/encoder.cc',
  'src/neural/factory.cc',
  'src/neural/loader.cc',
  'src/neural/network_check.cc',
  'src/neural/network_demux.cc',
  'src/neural/network_legacy.cc',
  'src/neural/network_mux.cc',
  'src/neural/network_random.cc',
  'src/neural/network_record.cc',
  'src/neural/network_rr.cc',
  'src/neural/network_trivial.cc',
  'src/neural/onnx/adapters.cc',
  'src/neural/onnx/builder.cc',
  'src/neural/onnx/converter.cc',
  'src/selfplay/game.cc',
  'src/selfplay/loop.cc',
  'src/selfplay/tournament.cc',
  'src/syzygy/syzygy.cc',
  'src/trainingdata/reader.cc',
  'src/trainingdata/trainingdata.cc',
  'src/trainingdata/writer.cc',
  'src/utils/commandline.cc',
  'src/utils/configfile.cc',
  'src/utils/esc_codes.cc',
  'src/utils/files.cc',
  'src/utils/histogram.cc',
  'src/utils/logging.cc',
  'src/utils/numa.cc',
  'src/utils/optionsdict.cc',
  'src/utils/optionsparser.cc',
  'src/utils/random.cc',
  'src/utils/string.cc',
  'src/utils/weights_adapter.cc',
  'src/version.cc',
]

deps += dependency('zlib', fallback: ['zlib', 'zlib_dep'])
deps += cc.find_library('pthread', required: true) 

files += 'src/utils/filesystem.posix.cc'
includes += include_directories('src')


if not get_option('popcnt')
  add_project_arguments('-DNO_POPCNT', language : 'cpp')
endif

if not get_option('pext')
  add_project_arguments('-DNO_PEXT', language : 'cpp')
endif

if get_option('embed')
  add_project_arguments('-DEMBED', language : 'cpp')
endif

if get_option('USE_SYCL')
      mlink_args = ['-fsycl']
      has_backends = true  
      message('Building SYCL')
      
      files += 'src/neural/sycl/layers.cc.dp.cpp'
      files += 'src/neural/sycl/network_sycl.cc.dp.cpp'
      files += 'src/neural/sycl/common_kernels.dp.cpp'


      DEF_INTEL_GENERAL_CXX_FLAGS = ['-O3','-fsycl','-ffast-math','-fsycl-unnamed-lambda','-Wall', '-Wextra'] 
      DEF_INTEL_WL_CXX_FLAGS = ['-DDEFAULT_MINIBATCH_SIZE=248', '-DMKL_ILP64']
      DEF_AMD_GENERAL_CXX_FLAGS = ['-O3','-fsycl','-ffast-math','-fsycl-unnamed-lambda','-Wall', '-Wextra']  
      DEF_AMD_WL_CXX_FLAGS = ['-DUSE_HIPBLAS', '-DINLINE', '-D__HIP_PLATFORM_AMD__'] 
      DEF_NVIDIA_GENERAL_CXX_FLAGS = ['-O3','-fsycl','-ffast-math','-fsycl-unnamed-lambda','-Wall', '-Wextra'] 
      DEF_NVIDIA_WL_CXX_FLAGS=['-DUSE_CUBLAS', '-DINLINE', '-DNVIDIABE']


      if(get_option('CMAKE_CXX_FLAGS') != [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') != [])
        message('Both  CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together')
      elif(get_option('CMAKE_CXX_FLAGS')== [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') == [])
        message('Using DEFAULT compilation flags')
        INTEL_GPU_CXX_FLAGS = DEF_INTEL_GENERAL_CXX_FLAGS + DEF_INTEL_WL_CXX_FLAGS
        NVIDIA_GPU_CXX_FLAGS = DEF_NVIDIA_GENERAL_CXX_FLAGS + DEF_NVIDIA_WL_CXX_FLAGS
        AMD_GPU_CXX_FLAGS = DEF_AMD_GENERAL_CXX_FLAGS + DEF_AMD_WL_CXX_FLAGS
      elif(get_option('OVERRIDE_GENERAL_CXX_FLAGS') !=[])
        message('OVERRIDING GENERAL compilation flags')
        INTEL_GPU_CXX_FLAGS = get_option('OVERRIDE_GENERAL_CXX_FLAGS') + DEF_INTEL_WL_CXX_FLAGS
        NVIDIA_GPU_CXX_FLAGS = get_option('OVERRIDE_GENERAL_CXX_FLAGS') + DEF_NVIDIA_WL_CXX_FLAGS
        AMD_GPU_CXX_FLAGS = get_option('OVERRIDE_GENERAL_CXX_FLAGS') + DEF_AMD_WL_CXX_FLAGS
      elif(get_option('CMAKE_CXX_FLAGS') != [])
        message('OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags')
        INTEL_GPU_CXX_FLAGS = get_option('CMAKE_CXX_FLAGS')
        NVIDIA_GPU_CXX_FLAGS = get_option('CMAKE_CXX_FLAGS')
        AMD_GPU_CXX_FLAGS = get_option('CMAKE_CXX_FLAGS')
      endif

      INTEL_GPU_CXX_FLAGS += [get_option('GPU_AOT')]
      NVIDIA_GPU_CXX_FLAGS += ['-fsycl-targets=nvidia_gpu_sm_' +  get_option('USE_SM')] 
      AMD_GPU_CXX_FLAGS += ['-fsycl-targets=amd_gpu_gfx' +  get_option('USE_SM')]


      if(get_option('USE_L0_BACKEND') == true)
        message('Building SYCL for the L0 backend')
        add_project_arguments(INTEL_GPU_CXX_FLAGS, language : 'cpp')
        deps += cc.find_library('sycl', required: true)
        deps += cc.find_library('mkl_sycl', required: true)
        deps += cc.find_library('mkl_intel_ilp64', required: true)
        deps += cc.find_library('mkl_sequential', required: true)
        deps += cc.find_library('mkl_core', required: true)   
        deps += cc.find_library('OpenCL', required: true)  
        deps += cc.find_library('dl', required: true)
        deps += cc.find_library('m', required: true)
        mlink_args += INTEL_GPU_CXX_FLAGS
      elif (get_option('USE_AMD_BACKEND') == true)
        message('Building SYCL for AMD backend')
        add_project_arguments(AMD_GPU_CXX_FLAGS, language : 'cpp') 
        hip_blas = cc.find_library('hipblas', required: true)
        hip_dart = cc.find_library('amdhip64', required: true)
        deps += [hip_blas, hip_dart]
        deps += cc.find_library('sycl', required: true)
        mlink_args+= AMD_GPU_CXX_FLAGS      
      else
        message('Building SYCL for the NVIDIA backend')
        add_project_arguments(NVIDIA_GPU_CXX_FLAGS, language : 'cpp') 
        cu_blas = cc.find_library('cublas', required: true)
        cu_dart = cc.find_library('cudart', required: true)
        cu_da = cc.find_library('cuda', required: true)
        deps += [cu_blas, cu_dart, cu_da]
        deps += cc.find_library('sycl', required: true)
        deps += cc.find_library('pthread', required: true)
        mlink_args+= NVIDIA_GPU_CXX_FLAGS  
      endif 
    
      executable('lc0_sycl', 'src/main.cc', files, include_directories: includes, dependencies: deps, install: true, link_args : mlink_args)

elif get_option('USE_CUDA')

  DEF_WL_CXX_FLAGS = ['-Xcompiler', '-fPIC']
  DEF_GENERAL_CXX_FLAGS = ['-O2']
  DEF_COMBINED_CXX_FLAGS = DEF_WL_CXX_FLAGS + DEF_GENERAL_CXX_FLAGS

  if(get_option('CMAKE_CXX_FLAGS') != [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') != [])
      message('Both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together')
  elif(get_option('CMAKE_CXX_FLAGS')== [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') == [])
      message('Using DEFAULT compilation flags')
      CMAKE_CXX_FLAGS = DEF_COMBINED_CXX_FLAGS
  elif(get_option('OVERRIDE_GENERAL_CXX_FLAGS') !=[])
      message('OVERRIDING GENERAL compilation flags')
      CMAKE_CXX_FLAGS = get_option('OVERRIDE_GENERAL_CXX_FLAGS') + DEF_WL_CXX_FLAGS
  elif(get_option('CMAKE_CXX_FLAGS') != [])
      message('OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags')
  endif


  cudnn_libdirs = get_option('cudnn_libdirs')
    cu_blas = cc.find_library('cublas', dirs: cudnn_libdirs, required: false)
    cu_dnn = cc.find_library('cudnn', dirs: cudnn_libdirs, required: false)
    cu_dart = cc.find_library('cudart', dirs: cudnn_libdirs, required: false)
    nvcc = find_program('nvcc', '/usr/local/cuda/bin/nvcc', '/opt/cuda/bin/nvcc', required: false)

    if (get_option('cudnn') or get_option('plain_cuda')) and cu_blas.found() and cu_dart.found() and nvcc.found()
      deps += [cu_blas, cu_dart]
      cuda_files = ['src/neural/cuda/layers.cc']
      if get_option('cudnn') and cu_dnn.found()
        deps += cu_dnn
        cuda_files += 'src/neural/cuda/network_cudnn.cc'
        add_project_arguments('-DUSE_CUDNN', language : 'cpp')
      endif
      if get_option('plain_cuda')
        cuda_files += 'src/neural/cuda/network_cuda.cc'
      endif
      foreach d : get_option('cudnn_include')
        if run_command('scripts/checkdir.py', d).returncode() == 0
          includes += include_directories(d)
        endif
      endforeach
      includes += include_directories('src/neural/cuda/')

      cuda_arguments = ['-c', '@INPUT@', '-o', '@OUTPUT@',
                        '-I', meson.current_source_dir() + '/src']
      if host_machine.system() == 'windows'
        if get_option('b_vscrt') == 'mt'
          cuda_arguments += ['-Xcompiler', '-MT']
        elif get_option('b_vscrt') == 'mtd'
          cuda_arguments += ['-Xcompiler', '-MTd']
        elif get_option('b_vscrt') == 'mdd' or (get_option('b_vscrt') == 'from_buildtype' and get_option('buildtype') == 'debug')
          cuda_arguments += ['-Xcompiler', '-MDd']
        elif get_option('b_vscrt') != 'none'
          cuda_arguments += ['-Xcompiler', '-MD']
        endif
      else
        cuda_arguments += CMAKE_CXX_FLAGS
        #cuda_arguments += ['--std=c++14', '-Xcompiler', '-fPIC']
      endif
      if get_option('nvcc_ccbin') != ''
        cuda_arguments += ['-ccbin=' + get_option('nvcc_ccbin')]
      endif
      cuda_cc = get_option('cc_cuda') # Unfortunately option cuda_cc is reserved.
      nvcc_extra_args = []
      if cuda_cc != ''
        nvcc_extra_args = ['-arch=compute_' + cuda_cc, '-code=sm_' + cuda_cc]
      endif
      foreach x : get_option('cudnn_include')
        cuda_arguments += ['-I', x]
      endforeach
      if host_machine.system() == 'windows'
        outputname = '@BASENAME@.obj'
      else
        outputname = '@BASENAME@.o'
      endif
      nvcc_extra_args += get_option('CUDA_NVCC_FLAGS')
      files += cuda_files
      files += custom_target('cuda fp32 code',
        input : 'src/neural/cuda/common_kernels.cu',
        output : outputname,
        depend_files: 'src/neural/cuda/winograd_helper.inc',
        command : [nvcc, nvcc_extra_args, cuda_arguments]
      )

      # Handling of fp16 cuda code.
      #nvcc_arch = '-arch=compute_' + get_option('USE_SM')
      nvcc_sm_list = ['sm_' + get_option('USE_SM')]
      # Ignore the given CC for fp16 when it is not in the supported list.
      if cuda_cc == '' or not nvcc_sm_list.contains('sm_' + cuda_cc)
        nvcc_extra_args = []
        nvcc_help = run_command(nvcc, '-h').stdout()
        foreach x : nvcc_sm_list
          if nvcc_help.contains(x)
            nvcc_extra_args += '-arch=' + x
          endif
        endforeach
      endif
      nvcc_extra_args += get_option('CUDA_NVCC_FLAGS')
      files += custom_target('cuda fp16 code',
        input : 'src/neural/cuda/fp16_kernels.cu',
        output : outputname,
        depend_files: 'src/neural/cuda/winograd_helper.inc',
        command : [nvcc, nvcc_extra_args, cuda_arguments]
      )
      has_backends = true
    endif

    executable('lc0_cuda', 'src/main.cc', files, include_directories: includes, dependencies: deps, install: true)

elif get_option('USE_AMD')
  files += 'src/neural/amd/layers.cpp'
  files += 'src/neural/amd/network_amd.cpp'
  files += 'src/neural/amd/common_kernels.cpp'

  DEF_WL_CXX_FLAGS = ['-D__HIP_PLATFORM_AMD__']
  DEF_GENERAL_CXX_FLAGS = ['-O3']
  DEF_COMBINED_CXX_FLAGS = DEF_WL_CXX_FLAGS + DEF_GENERAL_CXX_FLAGS

  if(get_option('CMAKE_CXX_FLAGS') != [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') != [])
      message('Both CMAKE_CXX_FLAGS and OVERRIDE_GENERAL_CXX_FLAGS cannot be passed in together')
  elif(get_option('CMAKE_CXX_FLAGS')== [] and get_option('OVERRIDE_GENERAL_CXX_FLAGS') == [])
      message('Using DEFAULT compilation flags')
      CMAKE_CXX_FLAGS = DEF_COMBINED_CXX_FLAGS
  elif(get_option('OVERRIDE_GENERAL_CXX_FLAGS') != [])
      message('OVERRIDING GENERAL compilation flags')
      CMAKE_CXX_FLAGS = get_option('OVERRIDE_GENERAL_CXX_FLAGS') + DEF_WL_CXX_FLAGS
  elif(get_option('CMAKE_CXX_FLAGS') != [])
      message('OVERRIDING GENERAL and WORKLOAD SPECIFIC compilation flags')
  endif

  add_project_arguments(CMAKE_CXX_FLAGS, language : 'cpp')

  hip_blas_lib = cc.find_library('hipblas', required: true)
  hip_blas_runtime = cc.find_library('hipblas', required: true)
  deps += hip_blas_lib  

  executable('lc0_amd', 'src/main.cc', files, include_directories: includes, dependencies: deps, install: true)

else

   message('Error no build option specified.')

endif



  

