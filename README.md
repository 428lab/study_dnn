# ディープラーニング(深層学習)の学習

GPUが動作するDockerコンテナ上で動作確認しています。
```sh
docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash
```
以下のコマンドで必要なパッケージをインストールしてください。
```sh
pip install -r requirements.txt
```

## RNN
RNNの学習に関するソースコード
- [MNISTをRNNで学習](https://github.com/428lab/study_dnn)
