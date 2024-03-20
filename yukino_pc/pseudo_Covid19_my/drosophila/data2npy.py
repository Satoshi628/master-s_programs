
import os
from skimage import io,transform
import numpy as np

## original class
#0   -> membrane | (0deg)    
#32  -> membrane / (45deg)  
#64  -> membrane - (90deg)  
#96  -> membrane \ (135deg) 
#128 -> membrane "junction" 
#159 -> glia/extracellular
#191 -> mitochondria
#223 -> synapse
#255 -> intracellular


## Hotta lab 5class
## 5 class
# 0 : membrane
# 1 : mitochondria
# 2 : synapse
# 3 : glia/extracellular
# 4 : intracellular

## 4 class
# 0 : extra/intracellular
# 1 : membrane
# 2 : mitochondria
# 3 : synapse


#"""
# gray scale image 2 class  ## 5 class
for i in range(0, 20):
    raw = io.imread('../../../groundtruth-drosophila-vnc/stack1/raw/%02d.tif' % (i)).astype(np.float32)
    lbl = io.imread('../../../groundtruth-drosophila-vnc/stack1/labels/labels%08d.png' % (i))
    lbl[lbl ==   0] = 0 # membrane
    lbl[lbl ==  32] = 0 # membrane
    lbl[lbl ==  64] = 0 # membrane
    lbl[lbl ==  96] = 0 # membrane
    lbl[lbl == 128] = 0 # membrane
    lbl[lbl == 159] = 3 # glia/extracellular
    lbl[lbl == 191] = 1 # mitochondria
    lbl[lbl == 223] = 2 # synapse
    lbl[lbl == 255] = 4 # intracellular
    raw = np.expand_dims(raw, axis=2)
    lbl = np.expand_dims(lbl, axis=2)
    print(raw.shape, lbl.shape)
    np.save('../../../groundtruth-drosophila-vnc/stack1/raw/%02d'%(i), np.concatenate((raw, lbl), axis=2))
#"""

""" #Tukawa nai hou ga ii
# mask image 2 class  ## 4 class 
for i in range(0, 20):
    raw = io.imread('../../../groundtruth-drosophila-vnc/stack1/raw/%02d.tif' % (i)).astype(np.float32)

    mem = np.zeros(raw.shape).astype(int)
    mit = np.zeros(raw.shape).astype(int)
    syn = np.zeros(raw.shape).astype(int)

    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/membranes/%02d.png' % (i))
    mem[img >= 128] = 1
    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/mitochondria/%02d.png' % (i))
    mit[img >= 128] = 1
    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/synapses/%02d.png' % (i))
    syn[img >= 128] = 1
    
    lbl_ = mem*1 + mit*2 + syn*3
    raw = np.expand_dims(raw, axis=2)
    lbl_ = np.expand_dims(lbl_, axis=2)
    print(raw.shape, lbl_.shape)
    print(np.min(lbl_), np.max(lbl_))
    np.save('../../../groundtruth-drosophila-vnc/stack1/raw/%02d'%(i), np.concatenate((raw, lbl_), axis=2))

#"""
""" #Tukawa nai hou ga ii
#gray and mask image 2 class ## 4&5 class
for i in range(0, 20):
    raw = io.imread('../../../groundtruth-drosophila-vnc/stack1/raw/%02d.tif' % (i)).astype(np.float32)

    mem = np.zeros(raw.shape)
    mit = np.zeros(raw.shape)
    syn = np.zeros(raw.shape)

    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/membranes/%02d.png' % (i))
    mem[img >= 128] = 1
    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/mitochondria/%02d.png' % (i))
    mit[img >= 128] = 1
    img = io.imread('../../../groundtruth-drosophila-vnc/stack1/synapses/%02d.png' % (i))
    syn[img >= 128] = 1
    
    lbl_ = mem*1 + mit*2 + syn*3

    lbl = io.imread('../../../groundtruth-drosophila-vnc/stack1/labels/labels%08d.png' % (i))
    lbl[lbl ==   0] = 0
    lbl[lbl ==  32] = 0
    lbl[lbl ==  64] = 0
    lbl[lbl ==  96] = 0
    lbl[lbl == 128] = 0
    lbl[lbl == 159] = 3
    lbl[lbl == 191] = 1
    lbl[lbl == 223] = 2
    lbl[lbl == 255] = 4

    raw = np.expand_dims(raw, axis=2)
    lbl_ = np.expand_dims(lbl_, axis=2)
    lbl = np.expand_dims(lbl, axis=2)
    print(raw.shape, lbl_.shape, lbl.shape)
    np.save('../../../groundtruth-drosophila-vnc/stack1/raw/%02d'%(i), np.concatenate((raw, lbl_, lbl), axis=2))
#"""
