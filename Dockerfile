# 使用带有Miniconda的官方Docker镜像作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 将Conda环境文件复制到Docker镜像中
COPY environment.yml /app/environment.yml

# 使用environment.yml重建Conda环境
RUN conda env create -f /app/environment.yml

# 激活Conda环境
SHELL ["conda", "run", "-n", "MAE_C", "/bin/bash", "-c"]

# 将项目文件夹和数据集文件夹复制到Docker镜像中
COPY . /app

# 暴露端口（如果需要）
EXPOSE 9999

# 设置默认运行的命令或脚本
#CMD ["conda", "run", "-n", "MAE_C", "/bin/bash"]
