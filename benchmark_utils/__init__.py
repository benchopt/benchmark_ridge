import platform
import subprocess
import os

# libomp x86_64 is needed for snapML on macOS. Homebrew is needed for this
# install as well.

# Homebrew is installed and added to path to install libomp x86_64.

# Also, libomp needs some directories added to PATH to function correctly.


# Check if the OS is macOS
if platform.system() == 'Darwin':
    # Install and setup Homebrew
    subprocess.run(
        [
            'arch', '-x86_64', '/bin/bash', '-c',
            "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/"
            "HEAD/install.sh)"
        ],
        check=True,
        shell=True
    )

    # Add Homebrew to the shell environment
    homebrew_shellenv = subprocess.run(
        '/usr/local/bin/brew shellenv',
        check=True,
        capture_output=True,
        text=True,
        shell=True
    ).stdout
    with open(os.path.expanduser('~/.profile'), 'a') as profile_file:
        profile_file.write(f'eval "$({homebrew_shellenv})"\n')
    subprocess.run(
        f'eval "$({homebrew_shellenv})"',
        check=True,
        shell=True
    )

    # Install libomp
    subprocess.run(
        'arch -x86_64 /usr/local/bin/brew install libomp',
        check=True,
        shell=True
    )

    # Set environment variables
    with open(os.path.expanduser('~/.profile'), 'a') as profile_file:
        profile_file.write('export LDFLAGS="-L/usr/local/opt/libomp/lib"\n')
        profile_file.write('export CPPFLAGS="-I/usr/local/opt/libomp/include"'
                           '\n')
        profile_file.write(
            'export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/opt/libomp'
            '/lib\n')

    # Also export environment variables in the current session
    os.environ['LDFLAGS'] = '-L/usr/local/opt/libomp/lib'
    os.environ['CPPFLAGS'] = '-I/usr/local/opt/libomp/include'
    os.environ['DYLD_LIBRARY_PATH'] = os.environ.get(
        'DYLD_LIBRARY_PATH', '') + ':/usr/local/opt/libomp/lib'
