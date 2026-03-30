import os
import sys

def print_tree(directory, prefix="", output_file=None):
    """
    递归打印目录树结构
    :param directory: 要遍历的目录路径
    :param prefix: 每行输出的前缀（用于控制缩进）
    :param output_file: 可选的输出文件对象（默认输出到控制台）
    """
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"错误：'{directory}' 不是一个有效的目录", file=sys.stderr)
        return

    # 获取目录下的所有条目（文件和文件夹），并排序（目录优先，然后文件）
    entries = []
    try:
        entries = os.listdir(directory)
    except PermissionError:
        print(f"{prefix}[权限不足，无法读取]")
        return

    # 分离目录和文件，分别排序
    dirs = sorted([e for e in entries if os.path.isdir(os.path.join(directory, e))])
    files = sorted([e for e in entries if os.path.isfile(os.path.join(directory, e))])
    # 目录放在前面，文件放在后面（与tree命令默认一致）
    sorted_entries = dirs + files

    # 遍历所有条目
    for i, entry in enumerate(sorted_entries):
        # 判断是否为最后一个条目（用于决定连接线）
        is_last = (i == len(sorted_entries) - 1)
        connector = "└── " if is_last else "├── "

        # 打印当前条目
        line = prefix + connector + entry
        if output_file:
            print(line, file=output_file)
        else:
            print(line)

        # 如果是目录，递归进入，并调整前缀
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            print_tree(full_path, prefix + extension, output_file)

def main():
    # 如果命令行提供了参数，则使用第一个参数作为起始目录；否则使用当前目录
    start_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # 打印根目录名称
    abs_path = os.path.abspath(start_dir)
    print(abs_path)

    # 调用打印函数
    print_tree(abs_path)

if __name__ == "__main__":
    main()