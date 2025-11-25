# cuda_by_example
《GPU高性能编程 CUDA实战》(《CUDA By Example an Introduction to General -Purpose GPU Programming》)随书代码

**IDE：** Visual Studio 2019

**CUDA Version：** 11.1

**Base on：**[CodedK/CUDA-by-Example-source-code-for-the-book-s-examples](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-.git)


**学习过程中遇到的一些坑：**

1. 编译报错：无法打开文件“glut64.lib“
   解决方法：项目->属性->VC++目录->库目录，包含lib文件夹

2. 运行报错：由于找不到glut64.dll，无法继续执行代码。重新安装程序可能会解决此问题。
   解决方法：将bin/glut64.dll文件复制到Debug中

## Ubuntu Build Instructions

1.  **Install Dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-cuda-toolkit freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
    ```

2.  **Build:**
    Run the build script to compile all examples:
    ```bash
    chmod +x build_all.sh
    ./build_all.sh
    ```

3.  **Run:**
    The compiled binaries are located in the `bin` directory.
    ```bash
    ./bin/hello_world
    ```
