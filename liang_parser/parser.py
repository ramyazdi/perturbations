#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

from __future__ import division

import sys

def shell_input():
    try:
        while True:
            yield raw_input()
    except:
        return

logs = sys.stderr

import newstate
from collections import defaultdict

from model import Model
from deptree import DepTree, DepVal

from mytime import Mytime
mytime = Mytime()
import time # TODO : MERGE

from wvector import WVector

import gc

import gflags as flags
FLAGS=flags.FLAGS

flags.DEFINE_integer("beam", 8, "beam width", short_name="b")
flags.DEFINE_integer("leftbeam", 1000, "leftptrs beam width") # number of left items (predictors to be combined w/ current)
flags.DEFINE_integer("kbest", 0, "kbest", short_name="k")
flags.DEFINE_boolean("forest", False, "dump the forest")
flags.DEFINE_integer("debuglevel", 0, "debug level (0: no debug info, 1: brief, 2: detailed)", short_name="D")
flags.DEFINE_boolean("dp", True, "use dynamic programming (merging)")
flags.DEFINE_boolean("newbeam", False, "new dp beaming")

flags.DEFINE_boolean("merge", False, "new dp beaming")
flags.DEFINE_boolean("naive", False, "do not merge at all")

#flags.DEFINE_boolean("donotcarei", False, "left boundary i not included in signature (state equality)")

#TODO: feature reengineering + learning
#TODO: cube pruning
#TODO: forest generation (and k-best)

uniqstats = defaultdict(list)

class Parser(object):

    State = None

    def __init__(self, model=None, b=FLAGS.b):
        
        Parser.model = model
        Parser.b = b
        Parser.State = __import__("newstate").NewState
        Parser.State.model = model
        Parser.debuglevel = FLAGS.debuglevel
        Parser.merge = FLAGS.merge
        Parser.naive = FLAGS.naive

    def try_parse(self, sent, refseq=None, update=None):
        ''' returns myseq, mytree, goodfeats, badfeats; update=None means non-training'''

        self.State.setup()

        n = len(sent)

        num_steps = 2*n + int(FLAGS.final_step)
        beams = [None] * (num_steps)
        beams[0] = [self.State.initstate(sent)] # initial state

        self.nstates = 0  # space complexity
        self.nedges = 0 # time complexity
        self.nuniq = 0
        if Parser.debuglevel >=3 and update == "early":
            print >> logs, "gold positions: ",
        sim_gold_item = None
        last_diff = 0
        largest_diff = -1
        max_diff = 0
        max_pos = None
        largest_pos, latest_pos, early_update_pos = None, None, None # possible to be non-early
        for i in range(1, num_steps): # 2n-1 steps + 1 final step

            buf = []
            gold_item = None
            for old in beams[i-1]:
                for action in old.allowed_actions():
                    action_gold = refseq is not None and action == refseq[i-1]  # 0-based
                    for new in old.take(action, action_gold):
                        buf.append(new)
                        if new.gold:
                            gold_item = new
                            
            if sim_gold_item is not None: # and update in ("late", "max"):
                for new in sim_gold_item.take(refseq[i-1], True):
                    if new.gold:
                        sim_gold_item = new
                        break                            
                
            self.nedges += len(buf) # number of created states (correlates with running time)

            if Parser.debuglevel >= 3:
                print >> logs, "\n".join([x.__str__(top=True) for x in sorted(buf)]) # beams[i]])
                print >> logs

            buf = sorted(buf) #[:self.b]
            tmp = {}
            beams[i] = []
            for j, new in enumerate(buf): # from best to worst
                if not FLAGS.dp or new not in tmp:
                    tmp[new] = new  ## first time to eval feats
                    new.rank = len(beams[i])
                    beams[i].append(new)
                else:
                    tmp[new].mergewith(new) # tmp[new] is the better equivalent state
                    if update == "naive" or (Parser.merge and new.action == 0): # SHIFT
                        new.rank = -2 # don't early stop!
                        if new.gold:
                            tmp[new].gold = True
                    elif Parser.debuglevel >=1 and update == "early" and new.gold:
                        print >> logs, "GOLD at %d MERGED with %d! " % (j, tmp[new].rank),

                if not FLAGS.newbeam:
                    if j == self.b - 1:  # |non-uniq|=b
                        break
                elif len(tmp) == self.b: # |uniq|=b
                    break                    

            self.nstates += len(beams[i]) # number of survived states
            self.nuniq += len(tmp)            
            #uniqstats[i].append(len(tmp)) # global variable

            if Parser.debuglevel >= 3:
                print >> logs, "\n".join([x.__str__(top=True) for x in beams[i]])
                print >> logs

            target = beams[i][0] # update against

            if update == "earliest" and gold_item.rank != 0: # first position that gold is not top
                return None, target.all_actions(), 0, i

            if update in ("early", "naive"):
                if gold_item.rank == -1: # early stop
                    if Parser.debuglevel >=3: 
                        print >> logs
                    if Parser.debuglevel >=1: 
                        print >> logs, "failed at step %d (len %d): %s" % (i, n, target)
                    return None, target.all_actions(), 0, i #, (gold_item.features, target.features)  ## no tree, no score
                else:
                    if Parser.debuglevel >=3:                                 
                        print >> logs, "-%d-" % gold_item.rank,

            if update in ("late", "large", "largest", "latest", "max", "hybrid", "maxcon"):
                if sim_gold_item is None and gold_item.rank == -1: # first fall-off
                    sim_gold_item = gold_item
                    early_update_pos = i
                if sim_gold_item is not None:
                    diff = beams[i][0].score - sim_gold_item.score
                    if update == "late" and diff < 0: # fall-in after fall-off
                        return None, beams[i-1][0].all_actions(), 0, i-1 # last step is still valid violation
                    if update == "large" and diff < last_diff:
                        return None, beams[i-1][0].all_actions(), 0, i-1 # last step is the largest violation
                    if diff > largest_diff:
                        largest_diff = diff
                        largest_pos = i
                    if diff >= 0:
                        latest_pos = i # for latest
                    last_diff = diff

            if update in ["max", "maxcon"]: # max over both left (before early update) and right
                try:
                    mdiff = beams[i][0].score - gold_item.score                                        
                except:
                    mdiff = beams[i][0].score - sim_gold_item.score
                if mdiff >= max_diff:
                    max_diff = mdiff
                    max_pos = i                    
        
        if Parser.debuglevel >=1 and update == "early":
            print >> logs # after gold positions line
            if gold_item.rank == 0:
                print >> logs, "succeeded at step %d" % i
            else:
                print >> logs, "finished wrong at step %d" % i            

        if update == "latest" and latest_pos is not None:
            return None, beams[latest_pos][0].all_actions(), 0, latest_pos
        if update == "largest" and largest_pos is not None:
            return None, beams[largest_pos][0].all_actions(), 0, largest_pos
        if update == "max": #and max_pos != i:
            return None, beams[max_pos][0].all_actions(), 0, max_pos
        if update == "maxcon" and max_pos != i:
            # otherwise full update
            return None, beams[max_pos][0].all_actions(), 0, max_pos            
            
        goal = beams[-1][0]
        self.beams = beams

        if update == "hybrid" and early_update_pos is not None and diff < 0:
            return None, beams[early_update_pos][0].all_actions(), 0, early_update_pos

        return goal.tree(), goal.all_actions(), goal.score, i #, (gold_item.features, target.features)

    def stats(self):
        return (self.nstates, self.nedges, self.nuniq)

    def simulate(self, actions, sent, first_diff=0, actionfeats=None, c=1):
        '''simulate the result of a given sequence of actions'''

        self.State.sent = sent

        n = len(sent)
        state = self.State.initstate(sent) # initial state
        if actionfeats is None:
            actionfeats = Model.new_weights(value_class=None)

        for i, action in enumerate(actions, 1):

##            actionfeats += state.make_feats(action) ## has to be OLD STATE -- WHY?
            if i >= first_diff:
                feats = state.make_feats(action)

                actionfeats.iaddl(Model.names[action], feats, c)
##                w = actionfeats[Model.names[action]]
##                for feat in feats:
##                    w[feat] += c
##                    if c > 0: # +1
##                        w.get(feat).incr()
##                    else: # -1
##                        w.get(feat).decr()

            if action in state.allowed_actions():
                for new in state.take(action):
                    state = new
                    break
            else:
                print >> logs, "Error! BAD SEQUENCE!"
                break

        return state, actionfeats

    def dumpforest(self, id=0):
        print "sent.%d\t%s" % (id, " ".join(["%s/%s" % wt for wt in self.State.sent]))
        finalbeam = self.beams[-1]
        nodes = set()
        nodeids = set()
        for x in finalbeam:
            x.previsit(nodes, nodeids)
            
##        print sum([len(beam) for beam in self.beams])
        print len(nodes) + 1 # number of nodes

        cache = set()
        for x in finalbeam:
            x.postvisit(cache)        

        # final root node
        print "%d\t-1 [%d-%d]\t%d ||| " % (len(nodes)+1, 0, len(self.State.sent), len(finalbeam))
        for state in finalbeam:
            print "\t%d ||| 0=0" % state.nodeid

        print

    def forestoracle(self, reftree):

        reflinks = reftree.links()
        oracle = -1 # oracle could be zero

        for i, state in enumerate(self.beams[-1]):

            subeval, tree = state.besteval(reflinks)
            h = tree.headidx

            root = 1 if (h in reflinks and reflinks[h] == -1) else 0
            roottot = 1 if h in reflinks else 0 # root could be punc
            rooteval = DepVal(yes=root, tot=roottot) # root link
            
            if rooteval + subeval > oracle:
#                print i, rooteval, subeval
                oracle = rooteval + subeval
                oracletree = tree

        print >> logs, "oracle=", oracle, reftree.evaluate(oracletree)
#        print "oracle=", oracletree
        return oracle, oracletree
            

####################################################################
    
def main():
    if FLAGS.sim is not None:
        sequencefile = open(FLAGS.sim)


    parser = Parser(model, b=FLAGS.beam)

    #ram change
    #print >> logs, "memory usage before parsing: ", human(memory(start_mem))

    totalscore = 0
    totalstates = 0
    totaluniq = 0
    totaledges = 0
    totaltime = 0

    totalprec = DepVal()    
    totaloracle = DepVal()

    print >> logs, "gc.collect unreachable: %d" % gc.collect()

    if FLAGS.manual_gc:
        gc.disable()
    
    i = 0
    gctime = 0
    input_file = open(FLAGS.input_file, 'r').readlines()
    # for i, line in enumerate(shell_input(), 1):
    for i, line in enumerate(input_file, 1):
        line = line.strip()
        if FLAGS.manual_gc and i % FLAGS.gc == 0:
            print >> logs, "garbage collection...",
            tt = time.time()
            print >> logs, "gc.collect unreachable: %d" % gc.collect()
            tt = time.time() - tt
            print >> logs, "took %.1f seconds" % tt
            gctime += tt

        line = line.strip()
        if line[0]=="(":
            # input is a gold tree (so that we can evaluate)
            reftree = DepTree.parse(line)
            sentence = DepTree.sent # assigned in DepTree.parse()
        else:
            # input is word/tag list
            reftree = None
            sentence = [tuple(x.rsplit("/", 1)) for x in line.split()]   # split by default returns list            
            DepTree.sent = sentence

        if FLAGS.debuglevel >= 1:
            print >> logs, sentence
            print >> logs, reftree #reftree is the gold

        mytime.zero()
        
        if FLAGS.sim is not None: # simulation, not parsing
            actions = map(int, sequencefile.readline().split())
            goal, feats = parser.simulate(actions, sentence) #if model is None score=0
            print >> logs, feats
            score, tree = goal.score, goal.top()
            (nstates, nedges, nuniq) = (0, 0, 0)
        else:
            # real parsing
            if True: #FLAGS.earlystop:
                refseq = reftree.seq() if reftree is not None else None
                tree, myseq, score, _ = parser.try_parse(sentence, refseq, update=False)
                if FLAGS.early:
                    print >> logs, "ref=", refseq
                    print >> logs, "myt=", myseq

                    refseq = refseq[:len(myseq)] # truncate
                    _, reffeats = parser.simulate(refseq, sentence) 
                    _, myfeats = parser.simulate(myseq, sentence)
                    print >> logs, "+feats", reffeats
                    print >> logs, "-feats", myfeats
                    
                nstates, nedges, nuniq = parser.stats()
            else:
                goal = parser.parse(sentence)
                nstates, nedges, nuniq = parser.stats()

##        score, tree = goal.score, goal.top()
#        score, tree = mytree
            
        dtime = mytime.period()

        if not FLAGS.early and not FLAGS.profile:
            if FLAGS.forest:
                parser.dumpforest(i)
            elif FLAGS.output:
                if not FLAGS.kbest:
                    print tree
                else:

                    stuff = parser.beams[-1][:FLAGS.kbest]
                    print "(sent.%d\t%d)" % (i, len(stuff))
                    for state in stuff:
                        #print "%.2f\t%s" % (state.score, state.tree())
                        print "%s" % state.tree()
                    print
                    
            if FLAGS.oracle:
                oracle, oracletree = parser.forestoracle(reftree)
                totaloracle += oracle

        prec = DepTree.compare(tree, reftree) # OK if either is None

        searched = sum(x.derivation_count() for x in parser.beams[-1]) if FLAGS.forest else 0
        #ram change
        #print >> logs, "sent {i:-4} (len {l}):\tmodelcost= {c:.2f}\tprec= {p:.2%}"\
        #     "\tstates= {ns} (uniq {uq})\tedges= {ne}\ttime= {t:.3f}\tsearched= {sp}" \
        #     .format(i=i, l=len(sentence), c=score, p=prec.prec(), \
        #             ns=nstates, uq=nuniq, ne=nedges, t=dtime, sp=searched)
        if FLAGS.seq:
            actions = goal.all_actions()
            print >> logs, " ".join(actions)
            check = parser.simulate(actions, sentence, model) #if model is None score=0
            checkscore = check.score
            checktree = check.top()
            print >> logs, checktree
            checkprec = checktree.evaluate(reftree)
            print >> logs, "verify: tree:%s\tscore:%s\tprec:%s" % (tree == checktree, score == checkscore, prec == checkprec)
            # ram change
            #print >> logs, "sentence %-4d (len %d): modelcost= %.2lf\tprec= %.2lf\tstates= %d (uniq %d)\tedges= %d\ttime= %.3lf" % \
            #      (i, len(sentence), checkscore, checkprec.prec100(), nstates, nuniq, nedges, dtime)

        totalscore += score
        totalstates += nstates
        totaledges += nedges
        totaluniq += nuniq
        totaltime += dtime

        totalprec += prec

    if i == 0:
        print >> logs, "Error: empty input."
        sys.exit(1)

    if FLAGS.featscache:
        print >> logs, "feature constructions: tot= %d shared= %d (%.2f%%)" % (State.tot, State.shared, State.shared / State.tot * 100)

    print >> logs, "beam= {b}, avg {a} sents,\tmodelcost= {c:.2f}\tprec= {p:.2%}" \
          "\tstates= {ns:.1f} (uniq {uq:.1f})\tedges= {ne:.1f}\ttime= {t:.4f}\n{d:s}" \
          .format(b=FLAGS.b, a=i, c=totalscore/i, p=totalprec.prec(), 
                  ns=totalstates/i, uq=totaluniq/i, ne=totaledges/i, t=totaltime/i, 
                  d=totalprec.details())
    
    if FLAGS.uniqstat:
        for i in sorted(uniqstats):
            print >> logs, "%d\t%.1lf\t%d\t%d" % \
                  (i, sum(uniqstats[i]) / len(uniqstats[i]), \
                   min(uniqstats[i]), max(uniqstats[i]))

    if FLAGS.oracle:
        print >> logs, "oracle= ", totaloracle

    if FLAGS.manual_gc:
        print >> logs, "garbage collection took %.1f seconds" % gctime

    print >> logs, "memory usage after parsing: ", human(memory(start_mem))
    #if FLAGS.mydouble: ###ram changes
        #from mydouble import counts
        #print >> logs, "mydouble usage and freed: %d %d" % counts()


def get_noise_info():
    if FLAGS.noise:
        noise_info = {
            'method': FLAGS.noise_method,
            'mu': FLAGS.mu,
            'sigma': FLAGS.sigma,
            'noise_file_path': FLAGS.noise_file_path,
        }
        return noise_info
    return None





if __name__ == "__main__":

    from newstate import NewState as State
    # to test speed per sentence (for plotting) please turn on gc, set it to 50 or 100 (which turns off auto gc)
    flags.DEFINE_boolean("manual_gc", False, "manual garbage collection")
    flags.DEFINE_integer("gc", 100, "garbage collect every X sentences (if automatic gc is turned off)")
    flags.DEFINE_boolean("uniqstat", False, "print uniq states stat info")
    flags.DEFINE_boolean("seq", False, "print action sequence")
    flags.DEFINE_string("sim", None, "simulate action sequences from FILE", short_name="s")

    flags.DEFINE_boolean("profile", False, "profile")

    flags.DEFINE_boolean("output", True, "output parsed results (turn it off for timing data)")
    flags.DEFINE_boolean("early", False, "use early update")

    flags.DEFINE_string("fakemem", None, "read in a file to occupy memory")

    flags.DEFINE_boolean("noise", False, "will insert noise into feature vector")
    flags.DEFINE_string("noise_method", None, "a for add, m for multiply")
    flags.DEFINE_float("mu", None, "noise mean")
    flags.DEFINE_float("sigma", None, "noise sigma")
    flags.DEFINE_string("noise_file_path", None, "path to file with vector of noises")

    flags.DEFINE_string("input_file", None, "the file we want to parse")

    flags.DEFINE_string("output_file_path", None, "our parsed results")



    argv = FLAGS(sys.argv)

    from monitor import memory, human

    noise_info = get_noise_info()

    start_mem = memory()


    if FLAGS.fakemem:
        s = Model(FLAGS.fakemem)
        t = Model(FLAGS.fakemem)
        print >> logs, "memory usage after read in fake: ", human(memory(start_mem))

    if FLAGS.weights is None:
        if not FLAGS.sim:
            print >> logs, "Error: must specify a weights file" + str(FLAGS)
            sys.exit(1)
        else:
            model = None # can simulate w/o a model
    else:
        model = Model(FLAGS.weights, noise_info) #FLAGS.model, FLAGS.weights)

    if FLAGS.profile:
        import cProfile as profile
        profile.run('main()', '/tmp/a')
        import pstats
        p = pstats.Stats('/tmp/a')
        p.sort_stats('cumulative', 'time').print_stats(60)

    else:
        main()
