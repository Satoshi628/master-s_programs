#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
# None

#----- 自作モジュール -----#
# None


class Script_Step():
    Scripts = ["Data Download",
               "Library Import",
               "Parameter Control",
               "Define DataLader",
               "Creat CNN",
               "Train Test Process",
               "Grad CAM Process"]

    # Scriptに必須な処理
    need_script = {"Data Download": [],
                   "Library Import": ["Data Download"],
                   "Parameter Control": [],
                   "Define DataLader": ["Data Download", "Parameter Control", "Library Import"],
                   "Creat CNN": ["Library Import"],
                   "Train Test Process": ["Data Download", "Library Import", "Parameter Control", "Define DataLader", "Creat CNN"],
                   "Grad CAM Process": ["Data Download", "Library Import", "Parameter Control", "Define DataLader", "Creat CNN", "Train Test Process"]}

    # スクリプトの表示体裁用
    Scripts_text = {"Data Download": "google driveのマウント",
                    "Library Import": "ライブリのインポート",
                    "Parameter Control": "パラメータ調節",
                    "Define DataLader": "DataLoder",
                    "Creat CNN": "CNNモデルの構成",
                    "Train Test Process": "main関数",
                    "Grad CAM Process": "Grad-Camで確認"}

    def __init__(self):
        super().__init__()
        self.Now_Process = "Data Download"
        self.Process_count = {script: 0 for script in self.Scripts}
        self.Process_count["Data Download"] += 1

    def __call__(self, now_script):
        self.Now_Process = now_script
        self.Process_count[now_script] += 1

        # 必須処理を実行したかチェック
        exit_flag = False
        for need in self.need_script[now_script]:
            # 一回も実行されていないなら
            if self.Process_count[need] == 0:
                exit_flag = True
                print(f"{self.Scripts_text[need]}が更新されていません、実行してください。")

        if exit_flag:
            print("必要なScriptを実行してください")
            sys.exit(0)  # 正常終了

        # 今実行したScriptが必須対象となっている処理のカウントを0にする
        for script in self.Scripts:  # need_scriptすべてを調査
            if now_script in self.need_script[script]:
                self.Process_count[script] = 0


class Script_Step_All():
    Scripts = ["Data Download",
               "Library Import",
               "Define Function",
               "Parameter Control",
               "Define DataLader",
               "Creat CNN",
               "Train Test Process",
               "Grad CAM Process"]

    # Scriptに必須な処理
    need_script = {"Data Download": [],
                   "Library Import": ["Data Download"],
                   "Define Function": ["Data Download", "Library Import"],
                   "Parameter Control": [],
                   "Define DataLader": ["Data Download", "Define Function", "Parameter Control", "Library Import"],
                   "Creat CNN": ["Library Import", "Define Function"],
                   "Train Test Process": ["Data Download", "Library Import", "Define Function", "Parameter Control", "Define DataLader", "Creat CNN"],
                   "Grad CAM Process": ["Data Download", "Library Import","Define Function" , "Parameter Control", "Define DataLader", "Creat CNN", "Train Test Process"]}

    # スクリプトの表示体裁用
    Scripts_text = {"Data Download": "google driveのマウント",
                    "Library Import": "ライブリのインポート",
                    "Define Function": "関数定義",
                    "Parameter Control": "パラメータ調節",
                    "Define DataLader": "DataLoder",
                    "Creat CNN": "CNNモデルの構成",
                    "Train Test Process": "main関数",
                    "Grad CAM Process": "Grad-Camで確認"}

    def __init__(self):
        super().__init__()
        self.Now_Process = "Data Download"
        self.Process_count = {script: 0 for script in self.Scripts}
        self.Process_count["Data Download"] += 1

    def __call__(self, now_script):
        self.Now_Process = now_script
        self.Process_count[now_script] += 1

        # 必須処理を実行したかチェック
        exit_flag = False
        for need in self.need_script[now_script]:
            # 一回も実行されていないなら
            if self.Process_count[need] == 0:
                exit_flag = True
                print(f"{self.Scripts_text[need]}が更新されていません、実行してください。")

        if exit_flag:
            print("必要なScriptを実行してください")
            sys.exit(0)  # 正常終了

        # 今実行したScriptが必須対象となっている処理のカウントを0にする
        for script in self.Scripts:  # need_scriptすべてを調査
            if now_script in self.need_script[script]:
                self.Process_count[script] = 0
