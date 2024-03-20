import time


class SpeedTimer():
    def __init__(self):
        self._start_time = 0.
        self._end_time = 0.
        self.times = 0.
        self.num = 0
    
    def __enter__(self):
        self._start_time = time.time()
        #return self #with ~ as ~:を使う際は必要

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self.times += self._end_time - self._start_time
        self.num += 1

    def __call__(self):
        # time = self.times / max(1, self.num)
        time = self.times
        self._reset()
        return time

    def _reset(self):
        self._start_time = 0.
        self._end_time = 0.
        self.times = 0.
        self.num = 0


class SpeedTester():
    def __init__(self):
        self._time_dict = {}
    
    def __getitem__(self, key):
        if not key in self._time_dict:
            self._time_dict[key] = SpeedTimer()
        return self._time_dict[key]  # キーに対応する値を取得

    def __setitem__(self, key, value):
        raise NotImplementedError("This class does not support item assignment")

    def __delitem__(self, key):
        del self._time_dict[key]  # キーに対応する項目を削除
    
    def items(self):
        return self._time_dict.items()

    def __str__(self):
        text = []
        for key, Time in self.items():
            text.append(f"{key}:{Time():.2f}")
        return "\n".join(text)

if __name__ == "__main__":
    Tester = SpeedTester()
    with Tester["pre_process"]:
        """
        pre process
        """
        time.sleep(0.3)
    with Tester["main_process"]:
        """
        main process
        """
        time.sleep(0.2)
    with Tester["pre_process"]:
        """
        pre process
        """
        time.sleep(0.2)
    print(Tester)