#!/usr/bin/env python3
"""
System Compatibility Check for Orochi/mamba_ssm

This script checks your system specifications and determines if mamba_ssm
can be installed, and if not, suggests alternatives.
"""

import sys
import subprocess
import platform


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_python():
    """Check Python version."""
    print_section("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version too old. Need Python 3.8+")
        return False


def check_cuda():
    """Check CUDA availability and version."""
    print_section("CUDA Check")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print("\nGPU Information:")
            print(result.stdout)

            # Try to get CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"\nDetected: {line.strip()}")

            return True
        else:
            print("✗ nvidia-smi failed to run")
            return False

    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  This means either:")
        print("    - No NVIDIA GPU installed")
        print("    - NVIDIA drivers not installed")
        print("    - Running on CPU-only system")
        return False


def check_gcc():
    """Check GCC/compiler version."""
    print_section("Compiler Check")

    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ GCC installed: {version_line}")
            return True
        else:
            print("✗ GCC not working properly")
            return False
    except FileNotFoundError:
        print("✗ GCC not found")
        print("  mamba_ssm requires a C++ compiler to build")
        return False


def check_system_info():
    """Check system information."""
    print_section("System Information")

    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")


def check_pytorch():
    """Check if PyTorch is installed and has CUDA."""
    print_section("PyTorch Check")

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"  CUDA available in PyTorch: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version in PyTorch: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    - Compute Capability: {props.major}.{props.minor}")
                print(f"    - Total Memory: {props.total_memory / 1e9:.2f} GB")
            return True
        else:
            print("✗ PyTorch installed but CUDA not available")
            return False

    except ImportError:
        print("✗ PyTorch not installed")
        return False


def suggest_alternatives(has_gpu, has_cuda, has_compiler):
    """Suggest alternatives based on system capabilities."""
    print_section("Recommendations & Alternatives")

    if has_gpu and has_cuda and has_compiler:
        print("✓ Your system appears compatible with mamba_ssm!")
        print("\nTry installing with:")
        print("  pip install mamba-ssm==2.2.2")
        print("  pip install causal-conv1d==1.4.0")
        print("\nIf installation still fails, please share the error message.")

    elif not has_gpu:
        print("✗ No NVIDIA GPU detected.")
        print("\n⚠️ CRITICAL: mamba_ssm REQUIRES an NVIDIA GPU with CUDA support.")
        print("   It cannot run on CPU-only systems.\n")

        print("📋 Your Options:\n")

        print("Option 1: Use Cloud Computing (RECOMMENDED)")
        print("-" * 70)
        print("A. Google Colab (FREE with GPU)")
        print("   - Visit: https://colab.research.google.com/")
        print("   - Free Tesla T4 GPU (limited hours)")
        print("   - Colab Pro: $9.99/month for better GPUs")
        print("   - Upload your code and run experiments there")

        print("\nB. Kaggle Notebooks (FREE with GPU)")
        print("   - Visit: https://www.kaggle.com/")
        print("   - Free 30 hours/week of GPU time")
        print("   - Tesla P100 GPUs available")

        print("\nC. AWS/GCP/Azure (PAID)")
        print("   - AWS EC2 with GPU instances (g4dn, p3, p4)")
        print("   - Google Cloud Compute with GPU")
        print("   - Azure GPU VMs")
        print("   - Costs: ~$0.50-$3.00 per hour depending on GPU")

        print("\nD. Lambda Labs (PAID - Good for ML)")
        print("   - Visit: https://lambdalabs.com/")
        print("   - GPU cloud specifically for ML")
        print("   - Competitive pricing")

        print("\n" + "-" * 70)
        print("Option 2: Use University/Institutional Resources")
        print("-" * 70)
        print("   - Check if your university has GPU clusters")
        print("   - Many universities offer free compute for students")
        print("   - Contact your IT department or research computing center")

        print("\n" + "-" * 70)
        print("Option 3: Get Access to a GPU System")
        print("-" * 70)
        print("   - Borrow a friend's gaming PC with NVIDIA GPU")
        print("   - Use a different workstation with GPU")
        print("   - Build/buy a system with NVIDIA GPU (long-term)")

        print("\n" + "-" * 70)
        print("Option 4: Alternative Models (NOT RECOMMENDED)")
        print("-" * 70)
        print("   - Orochi is specifically built on Mamba architecture")
        print("   - Cannot easily replace with other architectures")
        print("   - Would require significant code rewriting")
        print("   - Results would not match the paper")

    elif not has_cuda:
        print("✗ CUDA not properly installed or configured.")
        print("\nYour Options:")
        print("1. Install NVIDIA drivers:")
        print("   - Visit: https://www.nvidia.com/Download/index.aspx")
        print("   - Download and install drivers for your GPU")
        print("\n2. Install CUDA toolkit:")
        print("   - Visit: https://developer.nvidia.com/cuda-downloads")
        print("   - Install CUDA 11.8 or 12.1")
        print("\n3. After installation, reinstall PyTorch:")
        print("   - conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")

    elif not has_compiler:
        print("✗ No C++ compiler found.")
        print("\nInstall GCC/compiler:")

        if platform.system() == "Linux":
            print("  On Ubuntu/Debian:")
            print("    sudo apt-get update")
            print("    sudo apt-get install build-essential")
            print("\n  On CentOS/RHEL:")
            print("    sudo yum groupinstall 'Development Tools'")

        elif platform.system() == "Darwin":
            print("  On macOS:")
            print("    xcode-select --install")
            print("  Note: mamba_ssm may not support macOS GPUs (MPS)")

        elif platform.system() == "Windows":
            print("  On Windows:")
            print("    Install Visual Studio Build Tools")
            print("    Or install Visual Studio Community Edition")

    else:
        print("⚠️ System check inconclusive.")
        print("Please manually share the error you're getting when installing mamba_ssm")


def main():
    """Run all checks."""
    print_section("Orochi/mamba_ssm System Compatibility Check")
    print("This script will check if your system can run mamba_ssm\n")

    # Run checks
    check_system_info()
    has_python = check_python()
    has_compiler = check_gcc()
    has_gpu = check_cuda()
    has_pytorch_cuda = check_pytorch()

    # Suggest alternatives
    suggest_alternatives(has_gpu, has_pytorch_cuda, has_compiler)

    print_section("Summary")
    print(f"Python 3.8+:     {'✓' if has_python else '✗'}")
    print(f"NVIDIA GPU:      {'✓' if has_gpu else '✗'}")
    print(f"PyTorch + CUDA:  {'✓' if has_pytorch_cuda else '✗'}")
    print(f"C++ Compiler:    {'✓' if has_compiler else '✗'}")

    print("\n" + "=" * 70)
    if has_gpu and has_pytorch_cuda and has_compiler:
        print("Your system should be able to install mamba_ssm!")
        print("If you still have issues, please share the specific error message.")
    else:
        print("Your system cannot install mamba_ssm due to missing requirements.")
        print("See the recommendations above for alternative options.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during system check: {e}")
        import traceback
        traceback.print_exc()
