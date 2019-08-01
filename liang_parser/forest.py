#!/usr/bin/env python

''' class Forest is a collection of nodes, and is responsible for loading/dumping the forest.
    The real classes Node and Hyperedge are defined in node_and_hyperedge.py.

    N.B. do not remove sp for an already removed forest.
'''

# wsj_00.00       No , it was n't Black Monday .
# 199
# 1 DT [0-1]    0 ||| 12321=... 46456=...
# ...
# 6 NP [0-2]    1 ||| 21213=... 7987=...
#   1 4 ||| 0=-5.342
# ...

## N.B. node ID might be 123-45-67 where -x are subcats due to annotations.
    
import sys, os, re
import math
import time
import copy

#sys.path.append(os.environ["NEWCODE"])

#import mycode

logs = sys.stderr

from utility import getfile
#from tree import Tree

from svector import Vector as FVector
from node_and_hyperedge import Node, Hyperedge

print_merit = False
cache_same = False

from deptree import DepTree

import gflags as flags
FLAGS=flags.FLAGS

class Forest(object):
    ''' a collection of nodes '''

    def copy(self):
        '''must be deep!'''
        return copy.deepcopy(self)
        
    def size(self):
        ''' return (num_nodes, num_edges) pair '''
        return len(self.nodes), self.num_edges ##sum([len(node.edges) for node in self.nodes.values()])

    def __init__(self, num, sentence, cased_sent, tag=""):
        self.tag = tag
        self.num = num
        self.nodes = {}  ## id: node
        self.nodeorder = [] #node

        self.sent = sentence
        # a backup of cased sentence, since sent itself is going to be lowercased.
        self.cased_sent = cased_sent
        self.len = len(self.sent)

        ## magic formula in charniak parser; no diff. in reranking
        self.delta = self.len * math.log(600) / math.log(2) 
        self.cells = {}   # cells [(2,3)]...
        self.num_edges = 0

        self.weights = FVector("0=-1")

    def __len__(self):
        "sentence length"
        return self.len

    def add_node(self, node):
        self.nodes[node.iden] = node
        self.nodeorder.append(node)
        
        node.forest = self ## important backpointer!
        node.wrd_seq = self.sent[node.span[0]: node.span[1]]

    def rehash(self):
        ''' after pruning'''

        for i in xrange(self.len):
            for j in xrange(i+1, self.len+1):
                self.cells[(i,j)] = []
        
        self.num_edges = 0
        for node in self:
            self.cells[node.span].append(node)
            self.num_edges += len(node.edges)

    def clear_bests(self):
        for node in self:
            node.bestres = None

    def adjust_output(self, (sc, tr, fv)):
        ## delta is magic number in charniak's liang_parser
        ## subs[0]: remove TOP level
        return -sc, tr, fv #.subs[0]
    
    def bestparse(self, weights=FVector("0=-1"), adjust=False):
        self.clear_bests()

        res = self.root.bestparse(weights)
        if adjust:
            return self.adjust_output(res)
        else:
            return res

    def prep_kbest(self):
        self.bestparse()
        for node in self:
            node.prepare_kbest()
            
        return self.root.bestres[0]     

    def iterkbest(self, maxk, threshold):
        ''' (lazy) generator '''

        bestscore = self.prep_kbest()
        root = self.root
        for k in xrange(maxk):
            root.lazykbest(k+1)
            if root.fixed or threshold is not None and root.klist[k][0] > bestscore + threshold:
                break           
            else:
                ret = root.klist[k]
                # for psyco
                yield ret
        
    def lazykbest(self, k, sentid=0, threshold=None):

        basetime = time.time()

        bestscore = self.prep_kbest()
        
        self.root.lazykbest(k)

        if threshold is not None:
            for i, (sc, tr, fv) in enumerate(self.root.klist):
                if sc > bestscore + threshold:
                    self.root.klist = self.root.klist[:i]
                    break           

        print >> logs, "sent #%s, %d-best (%d uniq) computed in %.2lf secs" % \
              (sentid, min(k, len(self.root.klist)), min(k, len(self.root.kset)),
               time.time() - basetime)

    @staticmethod
    def load(filename, lower=False, sentid=0):
        '''now return a generator! use load().next() for singleton.
           and read the last line as the gold tree -- TODO: optional!
           and there is an empty line at the end
        '''

        file = getfile(filename)
        
        line = None
        total_time = 0
        num_sents = 0
        
        while True:         
            
            start_time = time.time()
            ##'\tThe complicated language in ...\n"
            ## tag is often missing
            try:
                if line is None or line == "\n":
                    line = "\n"
                    while line == "\n":
                        line = file.readline()  # emulate seek                  
                tag, sent = line.split("\t")
            except:
                ## no more forests
                break

            num_sents += 1
            
            sent = sent.split()
            cased_sent = sent [:]
            if lower:
                sent = [w.lower() for w in sent]   # mark johnson: lowercase all words
            num = int(file.readline())

            forest = Forest(num, sent, cased_sent, tag)
            forest.labelspans = {}
            forest.short_edges = {}

            delta = num_spu = 0
            for i in xrange(1, num+1):

                ## '2\tDT* [0-1]\t1 ||| 1232=2 ...\n'
                ## node-based features here: wordedges, greedyheavy, word(1), [word(2)], ...
                line = file.readline()
                try:
                    keys, fields = line.split(" ||| ")
                except:
                    keys = line
                    fields = ""


                iden, labelspan, size = keys.split("\t") ## iden can be non-ints
                size = int(size)

                fvector = FVector(fields) # TODO: myvector
                node = Node(iden, labelspan, size, fvector, sent)
                forest.add_node(node)

                if cache_same:
                    if labelspan in forest.labelspans:
                        node.same = forest.labelspans[labelspan]
                        node.fvector = node.same.fvector
                    else:
                        forest.labelspans[labelspan] = node

                for j in xrange(size):
                    is_oracle = False

                    ## '\t1 ||| 0=8.86276 1=2 3\n'
                    tails, fields = file.readline().strip().split(" ||| ")
                    
                    if tails[0] == "*":  #oracle edge
                        is_oracle = True
                        tails = tails[1:]
                        
                    tails = tails.split() ## could be non-integers
                    tailnodes = []

                    for x in tails:
                        assert x in forest.nodes, "BAD TOPOL ORDER: node #%s is referred to " % x + \
                               "(in a hyperedge of node #%s) before being defined" % iden
                        ## topological ordering
                        tail = forest.nodes[x]
                        tailnodes.append(tail)

                    use_same = False
                    if fields[-1] == "~":
                        use_same = True
                        fields = fields[:-1]
                        
                    fvector = FVector(fields)
                    edge = Hyperedge(node, tailnodes, fvector)

                    if cache_same:

                        short_edge = edge.shorter()
                        if short_edge in forest.short_edges:
                            edge.same = forest.short_edges[short_edge]
                            if use_same:
                                edge.fvector += edge.same.fvector
                        else:
                            forest.short_edges[short_edge] = edge

                    node.add_edge(edge)
                    if is_oracle:
                        node.oracle_edge = edge

                    
                if node.sp_terminal():
                    node.word = node.edges[0].subs[0].word

            ## splitted nodes 12-3-4 => (12, 3, 4)
            tmp = sorted([(map(int, x.iden.split("-")), x) for x in forest.nodeorder])   
            forest.nodeorder = [x for (_, x) in tmp]

            forest.rehash()
            sentid += 1
            
##          print >> logs, "sent #%d %s, %d words, %d nodes, %d edges, loaded in %.2lf secs" \
##                % (sentid, forest.tag, forest.len, num, forest.num_edges, time.time() - basetime)

            forest.root = node
            node.set_root(True)

            line = file.readline()

            if line is not None and line.strip() != "":
                if line[0] == "(":
##                    forest.goldtree = Tree.parse(line.strip(), trunc=True, lower=False)
                    line = file.readline()
            else:
                line = None

            total_time += time.time() - start_time

            if num_sents % 100 == 0:
                print >> logs, "... %d sents loaded (%.2lf secs per sent) ..." \
                      % (num_sents, total_time/num_sents)
                
            yield forest

        Forest.load_time = total_time
        if num_sents > 0:
            print >> logs, "%d forests loaded in %.2lf secs (avg %.2lf per sent)" \
                  % (num_sents, total_time, total_time/num_sents)

    @staticmethod
    def loadall(filename):
        forests = []
        for forest in Forest.load(filename):
            forests.append(forest)
        return forests

    def dump(self, out=sys.stdout):
        '''output to stdout'''
        # wsj_00.00       No , it was n't Black Monday .
        # 199
        # 1 DT [0-1]    0 ||| 12321=... 46456=...
        # ...
        # 6 NP [0-2]    1 ||| 21213=... 7987=...
        #   1 4 ||| 0=-5.342
        # ...

        if type(out) is str:
            out = open(out, "wt")

        # CAUTION! use original cased_sent!
        print >> out, "%s\t%s" % (self.tag, " ".join(self.cased_sent))
        print >> out, len(self.nodes)
        for node in self:

            oracle_edge = node.oracle_edge if hasattr(node, "oracle_edge") else None
            
            print >> out, "%s\t%d |||" % (node.labelspan(separator="\t"), len(node.edges)),
            if hasattr(node, "same"):
                print >> out, " "
            else:
                print >> out, node.fvector
                
            ##print >> out, "||| %.4lf" % node.merit if print_merit else ""

            for edge in node.edges:

                is_oracle = "*" if (edge is oracle_edge) else ""
                
                print >> out, "\t%s%s |||" % (is_oracle, " ".join([x.iden for x in edge.subs])),
                if cache_same and hasattr(edge, "same"):                    
                    diff = edge.fvector - edge.same.fvector
                    if 0 not in diff:
                        diff[0]=0
                    print >> out, diff, "~"
                else:
                    print >> out, edge.fvector
                    
                ##print >> out, "||| %.4lf" % edge.merit if print_merit else ""

        if hasattr(self, "goldtree"):
            print >> out, self.goldtree

        print >> out  ## last blank line

    def __iter__(self):
        for node in self.nodeorder:
            yield node

    def reverse(self):
        for i in range(len(self.nodeorder)):
            ret = self.nodeorder[-(i+1)]            
            yield ret

    def oracle(self, reftree):

        reflinks = reftree.links()
        oracle = 0

        for i, edge in enumerate(self.root.edges):
            node = edge.tails[0] # unary nodes
            h = node.headidx
            root = 1 if (h in reflinks and reflinks[h] == -1) else 0
            rooteval = DepVal(yes=root, tot=1) # root link

            subeval, tree = state.besteval(reflinks)
            
            if rooteval + subeval > oracle:
#                print i, rooteval, subeval
                oracle = rooteval + subeval
                oracletree = tree

        print >> logs, "oracle=", oracle, reftree.evaluate(oracletree)
#        print "oracle=", oracletree
        return oracle, oracletree
    
if __name__ == "__main__":

    try:
        import psyco
        psyco.full()
    except:
        pass

    flags.DEFINE_integer("k", 1, "k-best")
    flags.DEFINE_float("threshold", None, "threshold/margin")
    flags.DEFINE_boolean("infinite", False, "+\inf-best (infinitely long list)")
    flags.DEFINE_integer("id", 0, "sentence id")
    flags.DEFINE_string("first", None, "F:T, dump forests from F to T (inclusive) to stdout")
    flags.DEFINE_boolean("print_canonical", False, "print the canonical bit")
    
    argv = FLAGS(sys.argv)

    for i, f in enumerate(Forest.load("-")):

        DepTree.sent = [x.rsplit("/", 1) for x in f.sent]
##        print DepTree.sent

        if FLAGS.first is not None:
            if i+1 >= first:
                f.dump()
            if i+1 >= last:
                break
            continue

        if not FLAGS.infinite:
            if FLAGS.k is None:
                FLAGS.k = 1
            f.lazykbest(FLAGS.k, sentid=f.tag, threshold=FLAGS.threshold)
            if not FLAGS.canonical:
                print "%s\t%d" % (f.tag, len(f.root.klist))
            else:
                print "%s\t%d" % (f.tag, len([tr for sc, tr, fv in f.root.klist if tr.canonical]))
                
            for sc, tr, fv in f.root.klist:
                if FLAGS.print_canonical:
                    print "%.2lf\t%s\tcanonical=%s" % (-sc, tr, tr.canonical)
                else:
                    if tr.canonical or not FLAGS.canonical:
                        print "%.2lf\t%s" % (-sc, tr)

        else:
            if FLAGS.k is None:
                FLAGS.k = 100000 ## inf
            for res in f.iterkbest(FLAGS.k, threshold=FLAGS.threshold):
                print "%.2lf\t%s" % (f.adjust_output(res)[:2])

        print
