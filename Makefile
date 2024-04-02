# Makefile

RM=rm -f
CC=cc -O -Wall -Werror
CURL=curl
GZIP=gzip

LIBS=-lm

DATADIR=./data
MNIST_FILES= \
	$(DATADIR)/train-images-idx3-ubyte \
	$(DATADIR)/train-labels-idx1-ubyte \
	$(DATADIR)/t10k-images-idx3-ubyte \
	$(DATADIR)/t10k-labels-idx1-ubyte

all: test_rnn

clean:
	-$(RM) ./bnn ./train_test ./rnn *.o

get_mnist:
	-mkdir ./data
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/train-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/train-labels-idx1-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-images-idx3-ubyte
	-$(CURL) http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | \
		$(GZIP) -dc > ./data/t10k-labels-idx1-ubyte

test_bnn: ./bnn
	./bnn

train_mnist: ./train_test $(MNIST_FILES)
	./train_test $(MNIST_FILES) 1

test_mnist: ./train_test $(MNIST_FILES)
	./train_test $(MNIST_FILES) 2

test_rnn: ./rnn
	./rnn

./bnn: bnn.c
	$(CC) -o $@ $^ $(LIBS)

./mnist: train_test.c cnn.c
	$(CC) -o $@ $^ $(LIBS)

./rnn: rnn.c
	$(CC) -o $@ $^ $(LIBS)

train_test.c: cnn.h
cnn.c: cnn.h
