KGAT-SR
======
Hi! You are welcome to visit here!<br>
This repository is used to release the code of KGAT-SR, a newly proposed model for session-based recommendation by our research team. **KGAT-SR** stands for the **Knowledge-Enhanced Graph Attention Network for Session-based Recommendation**, which uses the knowledge about items from a KG to enhance session embedding via graph attention networks. To the best of our knowledge, it is the first session-based recommendation model that exploits external knowledge to enhance session embedding via graph attention networks. The research paper of KGAT-SR has been published in the 2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI), which is available at: https://doi.org/10.1109/ICTAI52525.2021.00164. The citation format in the IEEE Style is as follows:

Q. Zhang, Z. Xu, H. Liu and Y. Tang, "KGAT-SR: Knowledge-Enhanced Graph Attention Network for Session-based Recommendation," *2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)*, 2021, pp. 1026-1033, doi: 10.1109/ICTAI52525.2021.00164.

In essence, we used PyTorch to implement KGAT-SR based on FGNN [by R. Qiu et al., CIKM'19, https://doi.org/10.1145/3357384.3358010] and KGAT [by X. Wang  et al., KDD'19, https://doi.org/10.1145/3292500.3330989]. Our main modifications include: i) Python code of the KESG Generation layer was produced by modifying the attentive embedding propagation layers in KGAT; ii) The Session Embedding Generation layer was implemented by replacing the Readout function in FGNN with SAGPool [by J. Lee et al., ICML'19, http://proceedings.mlr.press/v97/lee19c.html].

Two real-world datasets (MovieLens 1M and LFM-1b) were used to empirically evaluate the performance of KGAT-SR, and the experimental results show that KGAT-SR significantly outperforms the state-of-the-art models for next item recommendation in terms of recommendation accuracy. Detailed information about the experimental datasets and the comparison models are given below.

Experimental Datasets
--
* **MovieLens 1M** (https://grouplens.org/datasets/movielens/1m/) [1]. This dataset contains user ratings for movies (i.e., interaction data) on the MovieLens website. Instead of the original dataset, our experiments directly used the preprocessed MovieLens 1M dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/rexrex9/kb4recMovielensDataProcess] by Yu et al.

* **LFM-1b** (http://www.cp.jku.at/datasets/LFM-1b/) [2]. This is a music dataset from the LFM-1b online system, describing users' interaction records on music. Instead of the original dataset, our experiments directly used the preprocessed LFM-1b dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/RUCDM/KB4Rec] by zhao et al [3].

References:

[1] Harper, F.M.; Konstan, J.A. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 2016, 5, 1-19. https://doi.org/10.1145/2827872

[2] Markus Schedl: The LFM-1b Dataset for Music Retrieval and Recommendation. ICMR 2016: 103-110. https://doi.org/10.1145/2911996.2912004

[3] Wayne Xin Zhao, Gaole He, Kunlin Yang, Hongjian Dou, Jin Huang, Siqi Ouyang, Ji-Rong Wen: KB4Rec: A Data Set for Linking Knowledge Bases with Recommender Systems. Data Intell. 1(2): 121-136 (2019). https://doi.org/10.1162/dint_a_00008

Comparison Models
--
* **BPR-MF** is a Bayesian personalized ranking (BPR) optimized matrix factorization (MF) model achieved by applying LearnBPR (a stochastic gradient descent algorithm based on bootstrap sampling) to MF.<br>
*`Paper:`* Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme: BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009: 452-461. http://arxiv.org/abs/1205.2618<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **FPMC** is the factorized personalized Markov chains model which combines both a common Markov chain and the normal matrix factorization model.<br>
*`Paper:`* Steffen Rendle, Christoph Freudenthaler, Lars Schmidt-Thieme: Factorizing personalized Markov chains for next-basket recommendation. WWW 2010: 811-820. https://doi.org/10.1145/1772690.1772773<br>
*`Code:`* https://github.com/khesui/FPMC

* **GRU4Rec** is an early and well-known sequential recommendation model based on RNN.<br>
*`Paper:`* Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk: Session-based Recommendations with Recurrent Neural Networks. ICLR (Poster) 2016. http://arxiv.org/abs/1511.06939<br>
*`Code:`* https://github.com/hidasib/GRU4Rec

* **STAMP** is a hybrid model that constructs two network structures to capture users’ general preferences and current interests.<br>
*`Paper:`* Qiao Liu, Yifu Zeng, Refuoe Mokhosi, Haibin Zhang: STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation. KDD 2018: 1831-1839. https://doi.org/10.1145/3219819.3219950<br>
*`Code:`* https://github.com/uestcnlp/STAMP

* **KSR** incorporates external knowledge into sequential recommender via key-value memory network.<br>
*`Paper:`* Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, Edward Y. Chang: Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. SIGIR 2018: 505-514. https://doi.org/10.1145/3209978.3210017<br>
*`Code:`* https://github.com/RUCDM/KSR

* **SR-GNN** represents the session sequence as a session graph and uses GNN to model complex transitions among items.<br>
*`Paper:`* Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, Tieniu Tan: Session-Based Recommendation with Graph Neural Networks. AAAI 2019: 346-353. https://doi.org/10.1609/aaai.v33i01.3301346<br>
*`Code:`* https://github.com/CRIPAC-DIG/SR-GNN

* **FGNN** collaboratively considers the sequential order and the latent order in the session graph, and formulates SR as a graph classification problem.<br>
*`Paper:`* Ruihong Qiu, Jingjing Li, Zi Huang, Hongzhi Yin: Rethinking the Item Order in Session-based Recommendation with Graph Neural Networks. CIKM 2019: 579-588. https://doi.org/10.1145/3357384.3358010<br>
*`Code:`* https://github.com/RuihongQiu/FGNN
