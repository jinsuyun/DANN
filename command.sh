
# svhn -> mnist
python main.py --source svhn --target mnist --gpus 1 --save svhn_mnist_tr1 > result/svhn_mnist_tr1.txt
python main.py --source svhn --target mnist --gpus 1 --save svhn_mnist_tr2 > result/svhn_mnist_tr2.txt
python main.py --source svhn --target mnist --gpus 1 --save svhn_mnist_tr3 > result/svhn_mnist_tr3.txt
python main.py --source svhn --target mnist --gpus 1 --save svhn_mnist_tr4 > result/svhn_mnist_tr4.txt
python main.py --source svhn --target mnist --gpus 1 --save svhn_mnist_tr5 > result/svhn_mnist_tr5.txt

# svhn -> mnist sumpooling width
#python main.py --source svhn --target mnist --gpus 0 --save "s->m_width_tr1" --sum_pooling width > result/s_m_width_tr1.txt &&
#python main.py --source svhn --target mnist --gpus 0 --save "s->m_width_tr2" --sum_pooling width > result/s_m_width_tr2.txt &&
#python main.py --source svhn --target mnist --gpus 0 --save "s->m_width_tr3" --sum_pooling width > result/s_m_width_tr3.txt &&
#python main.py --source svhn --target mnist --gpus 0 --save "s->m_width_tr4" --sum_pooling width > result/s_m_width_tr4.txt &&
#python main.py --source svhn --target mnist --gpus 0 --save "s->m_width_tr5" --sum_pooling width > result/s_m_width_tr5.txt

# svhn -> mnist sumpooling height
#python main.py --source svhn --target mnist --gpus 2 --save "s->m_height_tr1" --sum_pooling height > result/s_m_height_tr1.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save "s->m_height_tr2" --sum_pooling height > result/s_m_height_tr2.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save "s->m_height_tr3" --sum_pooling height > result/s_m_height_tr3.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save "s->m_height_tr4" --sum_pooling height > result/s_m_height_tr4.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save "s->m_height_tr5" --sum_pooling height > result/s_m_height_tr5.txt

#mnist -> mnist-m
#python main.py --gpus 0 --save mnist_mnistm_tr1 > result/mnist_mnistm_tr1.txt
#python main.py --gpus 0 --save mnist_mnistm_tr2 > result/mnist_mnistm_tr2.txt
#python main.py --gpus 0 --save mnist_mnistm_tr3 > result/mnist_mnistm_tr3.txt
#python main.py --gpus 0 --save mnist_mnistm_tr4 > result/mnist_mnistm_tr4.txt
#python main.py --gpus 0 --save mnist_mnistm_tr5 > result/mnist_mnistm_tr5.txt


