============================
Installing SnapML on macOS
============================

This tutorial provides step-by-step instructions to install SnapML on a macOS machine. The process includes setting up Homebrew and installing the necessary dependencies.

Steps to Install SnapML
========================

1. **Install and Setup Homebrew**

   Homebrew is a package manager for macOS that simplifies the installation of software.

   Open your terminal and run the following command to install Homebrew:

   .. code-block:: bash

      arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   After installing Homebrew, you need to add it to your shell environment:

   .. code-block:: bash

      echo 'eval "$(/usr/local/bin/brew shellenv)"' >> $HOME/.profile
      eval "$(/usr/local/bin/brew shellenv)"

2. **Install libomp**

   `libomp` is an OpenMP library required by SnapML. Install it using Homebrew:

   .. code-block:: bash

      arch -x86_64 /usr/local/bin/brew install libomp

   Once installed, set up the environment variables needed for `libomp`:

   .. code-block:: bash

      echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> $HOME/.profile
      echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> $HOME/.profile
      echo 'export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/opt/libomp/lib' >> $HOME/.profile

   Apply the changes by running:

   .. code-block:: bash

      source $HOME/.profile

3. **Install SnapML**

   With the dependencies installed, you can now proceed to install SnapML using pip:

   .. code-block:: bash

      pip install snapml


By following these steps, you should have SnapML installed and running correctly on your macOS machine.
