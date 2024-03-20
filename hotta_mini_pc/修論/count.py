import ast
from collections import Counter


# 変数の使用を訪問してカウントするクラス
class VariableCounter(ast.NodeVisitor):
    def __init__(self):
        self.counter = Counter()

    def visit_Name(self, node):
        self.counter[node.id] += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.counter['for'] += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.counter['function_def'] += 1
        self.generic_visit(node)

def process_files(file_list):
    """ファイルリスト内の各Pythonファイルについて変数の使用回数をカウントし、結果を表示します。"""
    
    # 変数のカウンターを使用して構文木を訪問
    counter = VariableCounter()

    for filename in file_list:
        # ファイルの読み込み
        with open(filename, "r",encoding="utf-8_sig") as file:
            file_content = file.read()
        
        # 構文木の解析
        tree = ast.parse(file_content, filename=filename)
        counter.visit(tree)
    print(counter.counter)

# 分析したいファイルのリスト
file_list = ["file1.py", "file2.py", "file3.py"]  # 実際には存在するファイルパスに置き換えてください

# ファイルリストの処理
process_files(file_list)