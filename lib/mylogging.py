from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG

loggers = {}

def mylogger(name=None, myfmt='%(asctime)s %(name)s %(lineno)d %(levelname)s %(message)s', shl='INFO', fpath='result_tmp/Input.log', fhl='DEBUG'):
	
	global loggers
	if name is None:
		name = __name__

	if loggers.get(name):
		return loggers.get(name)  #既出のロガーなら、ここで関数call終了

	# ログのフォーマットを設定
	fmt = Formatter(myfmt)

	# ログのコンソール出力の設定
	shandler = StreamHandler()
	shandler.setLevel(shl)
	shandler.setFormatter(fmt)

	# ログのファイル出力先の設定
	fhandler = FileHandler(fpath)
	fhandler.setLevel(fhl)
	fhandler.setFormatter(fmt)

	# ロガーの作成 
	logger = getLogger(__name__)  # ログの出力名を設定
	logger.setLevel(DEBUG)  #ログレベルの設定
	logger.addHandler(shandler)
	logger.addHandler(fhandler)
	logger.propagate = False
	loggers[name] = logger
	return logger


if __name__ == '__main__':
    test = mylogger()
