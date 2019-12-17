### Dependencies ###
NOTE: This code has been updated, if you were using this repo earlier and
experienced issues that was due to an outaded codebase. Please try again, and
if you're still stuck please send me an email: joey.bose@mail.mcgill.ca

Paper Link: https://arxiv.org/abs/1905.10674
1. Comet ML for logging. You will need an API key, username, and project name to do online logging.
2. Pytorch version=1.0
3. scikit-learn
4. tqdm for progress bar
5. pickle
6. json
7. joblib
8. networkx for creating reddit graph

To conduct experiments you will need to download the appropriate datasets and
preprocess them with the given preprocesssing scripts. This will involve
changing the file paths from their default ones. For FB15k-237 there is the
main dataset as well as the entity types dataset (links are provided in the
main paper). Further, note that reddit uses 2 steps of preprocessing,
the first to parse the json objects and then a second
one to create the K-core graph.

### Sample Commands ###
To reproduce the results we provide sample commands. Command Line arguments
control which sensitive attributes are use and whether there is a compositional
adversary or not.

1. FB15k-237:
`ipython --pdb -- paper_trans_e.py --namestr='FB15k Comp Gamma=1000' --do_log
--num_epochs=100 --embed_dim=20 --test_new_disc --sample_mask=True
--use_attr=True --gamma=1000 --valid_freq=50`

2. MovieLens1M:

`ipython --pdb -- main_movielens.py --namestr='100 GCMC Comp and Dummy'
--use_cross_entropy --num_epochs=200 --test_new_disc --use_1M=True
--show_tqdm=True --report_bias=True --valid_freq=5 --use_gcmc=True
--num_classifier_epochs=200 --embed_dim=30 --sample_mask=True --use_attr=True
--gamma=10 --do_log`

3. Reddit:

`ipython --pdb -- main_reddit.py --namestr='Reddit Compositional No Held Out
V2 Gamma=1' --valid_freq=5 --num_sensitive=10 --use_attr=True
--use_cross_entropy --test_new_disc --num_epochs=50 --num_nce=1
--sample_mask=True --debug --gamma=1000`

If you use this codebase or ideas in the paper please cite:

`@article{bose2019compositional,
  title={Compositional Fairness Constraints for Graph Embeddings},
  author={Bose, Avishek Joey and Hamilton, William},
  conference={Proceedings of the Thirty-sixth International Conference on Machine Learning, Long Beach CA},
  year={2019}
}`
