All:
	python main_fisher.py --dataset cifar10 --dataroot /home/iu_ai_club/gan_2018/FisherGAN/CIFAR_data/cifar-10-batches-py --niter 200 --cuda --Diters 2 --adam --lrG 2e-4 --lrD 2e-4 --imageSize 32 --G_extra_layers 2 --D_extra_layers 2

DeliFisher:
	python main_deli_fisher.py --dataset cifar10 --dataroot /home/miao/Documents/courses/cv/gan/FisherGAN/CIFAR_data/cifar-10-batches-py --niter 200 --cuda --Diters 2 --adam --lrG 2e-4 --lrD 2e-4 --imageSize 32 --G_extra_layers 2 --D_extra_layers 2

rm:
	rm ../../Inception-Score/data/*.*

mv:
	cp fake_samples_* ../../Inception-Score/data/ 

gen:
	rm samples/*.png
	python generateSamples.py --dataset cifar10 --dataroot /home/iu_ai_club/gan_2018/FisherGAN/CIFAR_data/cifar-10-batches-py --niter 200 --cuda --Diters 2 --adam --lrG 2e-4 --lrD 2e-4 --imageSize 32 --G_extra_layers 2 --D_extra_layers 2
	
 

