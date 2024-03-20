import os
import ast
import argparse
import shutil
from collections import defaultdict

import glob
import numpy as np


class SyntaxCounter(ast.NodeVisitor):
    def __init__(self):
        self.counts = defaultdict(int)
        self.class_count = 0
        self.import_count = 0
        self.assign_count = 0
        self.call_count = 0
        self.return_count = 0
        self.function_count = 0
        self.for_count = 0
        self.if_count = 0
    
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.import_count += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.import_count += 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.assign_count += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.call_count += 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.return_count += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node): #関数定義
        self.function_count += 1
        self.generic_visit(node) 

    def visit_For(self, node): #for文
        self.for_count += 1
        self.generic_visit(node)

    def visit_If(self, node): #if文
        self.if_count += 1
        self.generic_visit(node)

    def visit_Name(self, node): #変数
        self.counts[node.id] += 1
        self.generic_visit(node)
    
    def __str__(self):
        text = []

        text.append(f"クラス定義数:{self.class_count}")
        text.append(f"インポート回数:{self.import_count}")
        text.append(f"変数定義代入回数:{self.assign_count}")
        text.append(f"関数呼び出し回数:{self.call_count}")
        text.append(f"リターン文数:{self.return_count}")
        text.append(f"関数定義数:{self.function_count}")
        text.append(f"for文回数:{self.for_count}")
        text.append(f"if文回数:{self.if_count}")
        
        """
        names = list(self.counts.keys())
        nums = list(self.counts.values())
        idx = np.argsort(nums)[::-1]
        names = [names[i] for i in idx]
        nums = [nums[i] for i in idx]
        print("変数")
        for name, num in zip(names, nums):
            print(f"{name}:{num}")
        """
        return "\n".join(text)


def args():
    # 引数解析器を作成
    parser = argparse.ArgumentParser(description="このスクリプトの説明")

    # 引数を追加
    parser.add_argument("-p", "--source_path", type=str, help="パス指定")
    parser.add_argument("-s", "--save_path", type=str, help="セーブパス指定", default="./pyfiles")

    # 引数を解析
    args = parser.parse_args()
    return args

def collect_pyfile(args):
    pyfile_paths = sorted(glob.glob(os.path.join(args.source_path, "**","*.py"), recursive=True))
    print("number of pyfile:", len(pyfile_paths))

    for file_path in pyfile_paths:
        # 新しいディレクトリ構造のパスを生成
        new_path = file_path.replace(args.source_path, args.save_path)
        new_dir = os.path.dirname(new_path)
        
        # 必要に応じてディレクトリを作成
        os.makedirs(new_dir, exist_ok=True)
        
        # ファイルを新しい場所にコピー
        shutil.copy(file_path, new_path)

    print("complete")

def count_syntax_usage(args):
    """指定されたPythonファイル内の構文要素の使用回数をカウントします。"""
    #カレントディレクトリから取得
    pyfile_paths = sorted(glob.glob(os.path.join(".", "**","*.py"), recursive=True))
    
    counter = SyntaxCounter()
    
    line_count = []
    for file_path in pyfile_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        with open(file_path, 'r', encoding="utf-8") as file:
            lines = file.readlines()
    
        # 行数をカウント
        line_count.append(len(lines))
        tree = ast.parse(file_content, filename=file_path)
        counter.visit(tree)

    return pyfile_paths, line_count, counter


if __name__ == "__main__":
    args = args()
    #collect_pyfile(args)
    pyfile_paths, length, counter = count_syntax_usage(args)

    topk = 30
    
    print("行数合計:", sum(length))
    print(counter)
    
    length = np.array(length, dtype=np.int32)
    topk_idx = np.argsort(length)[::-1][:topk]
    print("行数ランキング")
    for i, idx in enumerate(topk_idx):
        print(f"第{i+1}位:{length[idx]}行\t:{pyfile_paths[idx]}")
    
    names = list(counter.counts.keys())
    nums = list(counter.counts.values())
    idx = np.argsort(nums)[::-1][:topk]
    names = [names[i] for i in idx]
    nums = [nums[i] for i in idx]
    print("変数ランキング")
    for idx, (name, num) in enumerate(zip(names, nums)):
        print(f"第{idx+1}位:{name}:{num}回")
    