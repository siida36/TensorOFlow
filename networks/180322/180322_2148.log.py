# coding: utf-8
from chainer import cuda
import simpleSeq2Seq
vocab_size =4000
model = simpleSeq2Seq.Seq2Seq(vocab_size=vocab_size, embed_size=128, hidden_size=128, batch_size=32, flag_gpu=1)
model.reset()
ARR = cuda.cupy
cuda.get_device(0).use()
model.to_gpu(0)
from chainer import optimizers
opt = optimizers.Adam()
opt.setup(model)
from chainer import optimizer
opt.add_hook(optimizer.GradientClipping(5))
total_loss = simpleSeq2Seq.forward([[1,0]*16],[[1,0]*16],model=model,ARR=ARR)
print(total_loss)
