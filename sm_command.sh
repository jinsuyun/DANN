# svhn -> mnist
#python main.py --source svhn --target mnist --gpus 1 --save sm_origin_tr1 > result/sm_origin_tr1.txt
#python main.py --source svhn --target mnist --gpus 1 --save sm_origin_tr2 > result/sm_origin_tr2.txt
#python main.py --source svhn --target mnist --gpus 1 --save sm_origin_tr3 > result/sm_origin_tr3.txt
#python main.py --source svhn --target mnist --gpus 1 --save sm_origin_tr4 > result/sm_origin_tr4.txt
#python main.py --source svhn --target mnist --gpus 1 --save sm_origin_tr5 > result/sm_origin_tr5.txt

# svhn -> mnist sumpooling width
#python main.py --source svhn --target mnist --gpus 2 --save sm_width_tr1 --sum_pooling width > result/sm_width_tr1.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_width_tr2 --sum_pooling width > result/sm_width_tr2.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_width_tr3 --sum_pooling width > result/sm_width_tr3.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_width_tr4 --sum_pooling width > result/sm_width_tr4.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_width_tr5 --sum_pooling width > result/sm_width_tr5.txt

# svhn -> mnist sumpooling height
#python main.py --source svhn --target mnist --gpus 2 --save sm_height_tr1 --sum_pooling height > result/sm_height_tr1.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_height_tr2 --sum_pooling height > result/sm_height_tr2.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_height_tr3 --sum_pooling height > result/sm_height_tr3.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_height_tr4 --sum_pooling height > result/sm_height_tr4.txt &&
#python main.py --source svhn --target mnist --gpus 2 --save sm_height_tr5 --sum_pooling height > result/sm_height_tr5.txt

#sumpooling matmul no cst
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save sm_matmul_origin_tr1 > result/sm_matmul_origin_tr1.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save sm_matmul_origin_tr2 > result/sm_matmul_origin_tr2.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save sm_matmul_origin_tr3 > result/sm_matmul_origin_tr3.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save sm_matmul_origin_tr4 > result/sm_matmul_origin_tr4.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save sm_matmul_origin_tr5 > result/sm_matmul_origin_tr5.txt

#sumpooling matmul cst encoder no sumpooling grl
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save s_m_sum_matmul_cst_encoder_nosumGRL_tr1 > result/s_m_sum_matmul_cst_encoder_nosumGRL_tr1.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save s_m_sum_matmul_cst_encoder_nosumGRL_tr2 > result/s_m_sum_matmul_cst_encoder_nosumGRL_tr2.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save s_m_sum_matmul_cst_encoder_nosumGRL_tr3 > result/s_m_sum_matmul_cst_encoder_nosumGRL_tr3.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save s_m_sum_matmul_cst_encoder_nosumGRL_tr4 > result/s_m_sum_matmul_cst_encoder_nosumGRL_tr4.txt
#python main.py --source svhn --target mnist --gpus 0 --sum_pooling both --save s_m_sum_matmul_cst_encoder_nosumGRL_tr5 > result/s_m_sum_matmul_cst_encoder_nosumGRL_tr5.txt


# svhn -> mnist sumpooling matmul cst disc no sumpooling grl
#python main.py --source svhn --target mnist --gpus 2 --save s_m_sum_matmul_cst_disc_nosumGRL_orig_tr1 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_nosumGRL_orig_tr1.txt
#python main.py --source svhn --target mnist --gpus 2 --save s_m_sum_matmul_cst_disc_nosumGRL_orig_tr2 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_nosumGRL_orig_tr2.txt
#python main.py --source svhn --target mnist --gpus 2 --save s_m_sum_matmul_cst_disc_nosumGRL_orig_tr3 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_nosumGRL_orig_tr3.txt
#python main.py --source svhn --target mnist --gpus 2 --save s_m_sum_matmul_cst_disc_nosumGRL_orig_tr4 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_nosumGRL_orig_tr4.txt
#python main.py --source svhn --target mnist --gpus 2 --save s_m_sum_matmul_cst_disc_nosumGRL_orig_tr5 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_nosumGRL_orig_tr5.txt

# svhn -> mnist sumpooling matmul cst disc
#python main.py --source svhn --target mnist --gpus 3 --save s_m_sum_matmul_cst_disc_orig_tr1 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_orig_tr1.txt
#python main.py --source svhn --target mnist --gpus 3 --save s_m_sum_matmul_cst_disc_orig_tr2 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_orig_tr2.txt
#python main.py --source svhn --target mnist --gpus 3 --save s_m_sum_matmul_cst_disc_orig_tr3 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_orig_tr3.txt
#python main.py --source svhn --target mnist --gpus 3 --save s_m_sum_matmul_cst_disc_orig_tr4 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_orig_tr4.txt
#python main.py --source svhn --target mnist --gpus 3 --save s_m_sum_matmul_cst_disc_orig_tr5 --cst --sum_pooling both > result/s_m_sum_matmul_cst_disc_orig_tr5.txt
