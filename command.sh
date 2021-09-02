
# svhn -> mnist
python main.py --source svhn --target mnist --gpus 1 --save "s->m_tr1" > result/s_m_tr1.txt &&
python main.py --source svhn --target mnist --gpus 1 --save "s->m_tr2" > result/s_m_tr2.txt &&
python main.py --source svhn --target mnist --gpus 1 --save "s->m_tr3" > result/s_m_tr3.txt &&
python main.py --source svhn --target mnist --gpus 1 --save "s->m_tr4" > result/s_m_tr4.txt &&
python main.py --source svhn --target mnist --gpus 1 --save "s->m_tr5" > result/s_m_tr5.txt

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
