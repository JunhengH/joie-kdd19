"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time
from KG import KG

class multiG(object):
    '''This class stores two KGs and alignment seeds. Initialize KGs separately because sometimes we don't load word embeddings
    '''

    def __init__(self, KG1=None, KG2=None):
        if KG1 == None or KG2 == None:
            self.KG1 = KG()
            self.KG2 = KG()
        else:
            self.KG1 = KG1
            self.KG2 = KG2
        self.lan1 = 'en'
        self.lan2 = 'fr'
        self.align = np.array([0])
        self.align_desc = np.array([0])
        self.aligned_KG1 = set([])
        self.aligned_KG2 = set([])
        self.aligned_KG1_index = np.array([0])
        self.aligned_KG2_index = np.array([0])
        self.unaligned_KG1_index = np.array([0])
        self.unaligned_KG2_index = np.array([0])
        self.align_valid = np.array([0])
        self.n_align = 0
        self.n_align_desc = 0
        self.ent12 = {}
        self.ent21 = {}
        self.batch_sizeK1 = 1024
        self.batch_sizeK2 = 64
        self.batch_sizeA = 32
        self.L1 = False
        self.dim1 = 300 #stored for TF_Part
        self.dim2 = 100
        #self.desc_dim = 64 #stored for TF_Part

    def load_align(self, filename, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n', desc=False):
        '''Load the dataset.'''
        weight = 1.
        align = []
        last_c = -1
        last_r = -1
        self.n_align = 0
        self.n_align_desc = 0
        self.align = []
        if desc:
            self.align_desc = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[2]) #change to triple
            if e1 == None or e2 == None:
                continue
            self.align.append((e1, e2))
            self.aligned_KG1.add(e1)
            self.aligned_KG2.add(e2)
            if self.ent12.get(e1) == None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) == None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            self.n_align += 1
            if desc:
                if (not self.KG1.get_desc_embed(e1) is None) and (not self.KG2.get_desc_embed(e2) is None):
                    self.align_desc.append((e1, e2))
                    self.n_align_desc += 1
        self.align = np.array(self.align)
        if desc:
            self.align_desc = np.array(self.align_desc)
        self.aligned_KG1_index = np.array([e for e in self.aligned_KG1])
        self.aligned_KG2_index = np.array([e for e in self.aligned_KG2])
        self.unaligned_KG1_index, self.unaligned_KG2_index = [], []
        for i in self.KG1.desc_index:
            if i not in self.aligned_KG1:
                self.unaligned_KG1_index.append(i)
        self.unaligned_KG1_index = np.array(self.unaligned_KG1_index)
        for i in self.KG2.desc_index:
            if i not in self.aligned_KG2:
                self.unaligned_KG2_index.append(i)
        self.unaligned_KG2_index = np.array(self.unaligned_KG2_index)
        print("Loaded aligned entities from", filename, ". #pairs:", self.n_align)

    def load_valid(self, filename, size=1024, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n', desc=False):
        '''Load the dataset.'''
        self.align_valid = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            if self.ent12.get(e1) == None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) == None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            if (not self.KG1.get_desc_embed(e1) is None) and (not self.KG2.get_desc_embed(e2) is None):
                self.align_valid.append((e1, e2))
                if len(self.align_valid) >= size:
                    break
        self.align_valid = np.array(self.align_valid)
        print("Loaded validation entities from", filename, ". #pairs:", size)

    def load_more_gt(self, filename):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            if self.ent12.get(e1) == None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) == None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            print("Loaded more gt file for negative sampling from", filename)

    def num_align(self):
        '''Returns number of entities. 

        This means all entities have index that 0 <= index < num_ents().
        '''
        return self.n_align
    
    def num_align_desc(self):
        '''Returns number of entities. 

        This means all entities have index that 0 <= index < num_ents().
        '''
        return self.n_align_desc
 
    def corrupt_desc_pos(self, align, pos, sample_global=True):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                if sample_global:
                    samp = np.random.choice(self.KG1.desc_index)
                else:
                    samp = np.random.choice(self.aligned_KG1_index)
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                if sample_global:
                    samp = np.random.choice(self.KG2.desc_index)
                else:
                    samp = np.random.choice(self.aligned_KG2_index)
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_desc(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_desc_pos(align, pos)
    
    def corrupt_align_pos(self, align, pos):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.randint(self.KG1.num_ents())
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.randint(self.KG2.num_ents())
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_align(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_align_pos(align, pos)
    
    #corrupt 
    def corrupt_desc_batch(self, a_batch, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_desc(a, tar) for a in a_batch])

    def corrupt_align_batch(self, a_batch, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_align(a, tar) for a in a_batch])
    
    def sample_false_pair(self, batch_sizeA):
        a = np.random.choice(self.unaligned_KG1_index, batch_sizeA)
        b = np.random.choice(self.unaligned_KG2_index, batch_sizeA)
        return np.array([(a[i], b[i]) for i in range(batch_sizeA)])
    
    def expand_align(self, list_of_pairs):
        # TODO
        pass
    
    def token_overlap(self, set1, set2):
        min_len = min(len(set1), len(set2))
        hit = 0.
        for tk in set1:
            for tk2 in set2:
                if tk == tk2:
                    hit += 1
        return hit / min_len

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)