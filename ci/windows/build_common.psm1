
Param(
    [Parameter(Mandatory = $true)]
    [Alias("cxx")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(11, 14, 17, 20)]
    [int]$CXX_STANDARD = 17
)


# We need the full path to cl because otherwise cmake will replace CMAKE_CXX_COMPILER with the full path
# and keep CMAKE_CUDA_HOST_COMPILER at "cl" which breaks our cmake script
$script:HOST_COMPILER  = (Get-Command "cl").source -replace '\\','/'
$script:PARALLEL_LEVEL = (Get-WmiObject -class Win32_processor).NumberOfLogicalProcessors

if (-not $env:CCCL_BUILD_INFIX) {
    $env:CCCL_BUILD_INFIX = ""
}

# Presets will be configured in this directory:
$BUILD_DIR = "../build/$env:CCCL_BUILD_INFIX"

If(!(test-path -PathType container "../build")) {
    New-Item -ItemType Directory -Path "../build"
}

# The most recent build will always be symlinked to cccl/build/latest
New-Item -ItemType Directory -Path "$BUILD_DIR" -Force

# replace sccache binary to get it working with MSVC
$script:path_to_sccache =(gcm sccache).Source
Remove-Item $path_to_sccache -Force
Invoke-WebRequest -Uri "https://github.com/robertmaynard/sccache/releases/download/nvcc_msvc_v1/sccache.exe" -OutFile $path_to_sccache

# Prepare environment for CMake:
$env:CMAKE_BUILD_PARALLEL_LEVEL = $PARALLEL_LEVEL
$env:CTEST_PARALLEL_LEVEL = 1
$env:CUDAHOSTCXX = $HOST_COMPILER.FullName
$env:CXX = $HOST_COMPILER.FullName

Write-Host "========================================"
Write-Host "Begin build"
Write-Host "pwd=$pwd"
Write-Host "BUILD_DIR=$BUILD_DIR"
Write-Host "CXX_STANDARD=$CXX_STANDARD"
Write-Host "CXX=$env:CXX"
Write-Host "CUDACXX=$env:CUDACXX"
Write-Host "CUDAHOSTCXX=$env:CUDAHOSTCXX"
Write-Host "NVCC_VERSION=$NVCC_VERSION"
Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"
Write-Host "CTEST_PARALLEL_LEVEL=$env:CTEST_PARALLEL_LEVEL"
Write-Host "CCCL_BUILD_INFIX=$env:CCCL_BUILD_INFIX"
Write-Host "Current commit is:"
Write-Host "$(git log -1)"
Write-Host "========================================"

function configure_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,
        [Parameter(Mandatory = $true)]
        [string]$CMAKE_OPTIONS
    )

    $step = "$BUILD_NAME (configure)"

    cmake --preset $PRESET $CMAKE_OPTIONS --log-level VERBOSE
    $test_result = $LastExitCode

    If ($test_result -ne 0) {
        throw '$step Failed'
    }

    Write-Host "$step complete."
}

function build_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    $step = "$BUILD_NAME (build)"

    sccache_stats('Start')

    cmake --build --preset $PRESET
    $test_result = $LastExitCode

    sccache_stats('Stop')

    echo "$step complete"
    
    If ($test_result -ne 0) {
         throw '$step Failed'
    }
}

function test_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET
    )

    $step = "$BUILD_NAME (test)"

    sccache_stats('Start')

    ctest --preset $PRESET
    $test_result = $LastExitCode

    sccache_stats('Stop')

    echo "$step complete"
    
    If ($test_result -ne 0) {
         throw '$step Failed'
    }
}

function configure_and_build_preset {
    Param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$BUILD_NAME,
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$PRESET,
        [Parameter(Mandatory = $true)]
        [string]$CMAKE_OPTIONS
    )

    configure_preset "$BUILD_NAME" "$PRESET" "$CMAKE_OPTIONS"
    build_preset "$BUILD_NAME" "$PRESET"
}

function sccache_stats {
    Param (
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [ValidateSet('Start','Stop')]
        [string]$MODE
    )

    $sccache_stats = sccache -s
    If($MODE -eq 'Start') {
        [int]$script:sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cpp   = ($sccache_stats[2] -replace '[^\d]+')
        [int]$script:sccache_cache_hits_cuda  = ($sccache_stats[3] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cpp   = ($sccache_stats[5] -replace '[^\d]+')
        [int]$script:sccache_cache_miss_cuda  = ($sccache_stats[6] -replace '[^\d]+')
    } else {
        [int]$final_sccache_compile_requests = ($sccache_stats[0] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cpp   = ($sccache_stats[2] -replace '[^\d]+')
        [int]$final_sccache_cache_hits_cuda  = ($sccache_stats[3] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cpp   = ($sccache_stats[5] -replace '[^\d]+')
        [int]$final_sccache_cache_miss_cuda  = ($sccache_stats[6] -replace '[^\d]+')

        [int]$total_requests  = $final_sccache_compile_requests - $script:sccache_compile_requests
        [int]$total_hits_cpp  = $final_sccache_cache_hits_cpp   - $script:sccache_cache_hits_cpp
        [int]$total_hits_cuda = $final_sccache_cache_hits_cuda  - $script:sccache_cache_hits_cuda
        [int]$total_miss_cpp  = $final_sccache_cache_miss_cpp   - $script:sccache_cache_miss_cpp
        [int]$total_miss_cuda = $final_sccache_cache_miss_cuda  - $script:sccache_cache_miss_cuda
        If ( $total_requests -gt 0 ) {
            [int]$hit_rate_cpp  = $total_hits_cpp  / $total_requests * 100;
            [int]$hit_rate_cuda = $total_hits_cuda / $total_requests * 100;
            echo "sccache hits cpp:  $total_hits_cpp  `t| misses: $total_miss_cpp  `t| hit rate: $hit_rate_cpp%"
            echo "sccache hits cuda: $total_hits_cuda `t| misses: $total_miss_cuda `t| hit rate: $hit_rate_cuda%"
        } else {
            echo "sccache stats: N/A No new compilation requests"
        }
    }
}

Export-ModuleMember -Function configure_preset, build_preset, test_preset, configure_and_build_preset, sccache_stats
Export-ModuleMember -Variable BUILD_DIR
