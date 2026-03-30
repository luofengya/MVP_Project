import os

def create_from_tree(file_path, root_dir='.'):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析缩进层级并构建路径
    stack = []  # 存储当前路径的各部分
    for line in lines:
        stripped = line.rstrip('\n')
        if not stripped or stripped.isspace():
            continue

        # 计算缩进级别（假设每级缩进4个空格）
        indent = len(line) - len(line.lstrip())
        level = indent // 4

        # 调整栈大小
        while len(stack) > level:
            stack.pop()

        # 获取当前项目名称（去除缩进和末尾的'/')
        name = line.strip().rstrip('/')
        if not name:
            continue

        # 构建完整路径
        current_path = os.path.join(root_dir, *stack, name)
        # 判断是否为目录（原行以'/'结尾）
        if line.strip().endswith('/'):
            os.makedirs(current_path, exist_ok=True)
            print(f"创建目录: {current_path}")
            stack.append(name)  # 将当前目录入栈
        else:
            # 创建文件（空文件）
            os.makedirs(os.path.dirname(current_path), exist_ok=True)
            with open(current_path, 'w') as f:
                pass
            print(f"创建文件: {current_path}")

if __name__ == '__main__':
    create_from_tree('structure.txt')