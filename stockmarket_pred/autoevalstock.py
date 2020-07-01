from event_to_embeddings import *
from event_stock_align import *
from train import *
model=['NTN','LowRankNTN']
actv=['sigmoid','relu','tanh']
aff=[40,30,20,10]
if __name__ == '__main__':
    #main('LowRankNTN','/users4/bwchen/CommonsenseERL_EMNLP_2019/testmodel/New2/New2Lowrank20relu_19',20)
    #main1()
    main2('LowRankNTN','/users4/bwchen/CommonsenseERL_EMNLP_2019/testmodel/New2/New2Lowrank20relu_19',20)
    # for name in model:
        # model=name
        # if name=='NTN':
            # for act in actv:
                # for i in range(1,21):
                    # model_file='/users4/bwchen/CommonsenseERL_EMNLP_2019/testmodel/'+'New5'+model+act+'_'+str(i)
                    # main(model, model_file, 40)
                    # main1()
                    # main2(model, model_file, 40)
        # else:
            # for rank in aff:
                # em_r = rank
                # for act in actv:
                    # for i in range(1,21):
                        # model=name
                        # model_file='/users4/bwchen/CommonsenseERL_EMNLP_2019/testmodel/'+'New5'+'Lowrank'+str(rank)+act+'_'+str(i)
                        # main(model, model_file, em_r)
                        # main1()
                        # main2(model, model_file, em_r)
