# all_code

Multiple projects for face emotion recognition:

google_fer_siamese: Project for fetching, training and testing google's dataset https://research.google/tools/datasets/google-facial-expression/
                    It uses multiprocessing to fetch the dataset faster. I am trying to train siamese network with a triplet loss.
                    To compute face feature pretrained resnet trained on vggface2 is used pipelined into a densenet and then to a fc layer.
                    
simple_net_fer: Project for face emotion recognition on dataset affectnet and ck+. 
