# Online Multi-Object Tracking with Dual Matching Attention Networks

This is the implementation of our ECCV2018 paper "Online Multi-Object Tracking with Dual Matching Attention Networks". We integrate the ECO [1] for single object tracking. The code framework for MOT benefits from the MDP [2].

# Prerequisites
- python 2.7
- Keras 2.0.5
- Tensorflow 1.1.0


# Usage
1. Download the [DMAN model](http) and put it into the "model/" folder.
2. Run the socket server script:
<pre><code>python calculate_similarity.py
</code></pre>
3. Run the socket client script DMAN_demo.m in Matlab.
# Citation

If you use this code, please consider citing:

<pre><code>@inproceedings{zhu-eccv18-DMAN,
    author    = {Zhu, Ji and Yang, Hua and Liu, Nian and Kim, Minyoung and Zhang, Wenjun and Yang, Ming-Hsuan},
    title     = {Online Multi-Object Tracking with Dual Matching Attention Networks},
    booktitle = {European Computer Vision Conference},
    year      = {2018},
}
</code></pre>

# References
[1] Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: ECO: Efficient convolution operators for tracking. In: CVPR (2017)

[2] Xiang, Y., Alahi, A., Savarese, S.: Learning to track: Online multi-object tracking by decision making. In: ICCV (2015)
