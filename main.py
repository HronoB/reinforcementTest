import math
import random
import time
from typing import List

import numpy as np

from stageController import Controller



NUMBER_OF_TARGETS = 2
class Stage:


    class FlatObject:
        def __init__(self, seed=0):
            self.x = 0.5
            self.y = 0.5
            self.accels = 3
            self.vx = [0]*self.accels
            self.vy = [0]*self.accels
            self.radius = 0.05
            self.dead = False

        def collide(self, other: Stage.FlatObject):
            if (self.x-other.x)**2+(self.y-other.y)**2 < (self.radius+other.radius)**2:
                return 1
            return 0

        def move(self, delta):
            self.x += self.vx[0]*delta
            self.y += self.vy[0]*delta
            for i in range(self.accels-1):
                self.vx[i]+=self.vx[i+1]*delta
                self.vy[i]+=self.vy[i+1]*delta
            for i in range(self.accels):
                decc = delta*0.5/math.factorial(i+1)
                self.vx[i] /= math.exp(decc)
                self.vy[i] /= math.exp(decc)

        def accel(self, order, vx, vy):
            self.vx[order] += vx/math.factorial(order+1)
            self.vy[order] += vy/math.factorial(order+1)

        def reset(self):
            self.x = 0.5
            self.y = 0.5
            self.vx = [0]*self.accels
            self.vy = [0]*self.accels


    def __init__(self):
        self.player = Stage.FlatObject()
        self.targets = []
        pass

    def getClosestObst(self):
        closest = self.targets[0]
        for i in self.targets:
            if (self.player.x-i.x)**2+(self.player.y-i.y)**2 < (self.player.x-closest.x)**2+(self.player.y-closest.y)**2:
                closest = i
        return closest



    def move(self, vec2):
        self.player.accel(1, vec2[0], vec2[1])


    def collide(self) -> int:
        result = 0
        for i in self.targets:
            score = self.player.collide(i)
            if score > 0:
                result += score
                i.dead = True
        return result


    def populate(self):
        ix = 0
        while ix < len(self.targets):
            if self.targets[ix].dead:
                self.targets[ix] = self.targets[-1]
                self.targets.pop()
            else:
                ix+=1
        while len(self.targets)<NUMBER_OF_TARGETS:
            self.targets.append(Stage.FlatObject())
            self.targets[-1].x = random.random()
            self.targets[-1].y = random.random()

    def tick(self, deltaTime = 1.0):
        self.player.move(deltaTime)
        self.populate()


    def getData(self):
        self.targets.sort(key=(lambda i: (i.x-self.player.x)**2+(i.y-self.player.y)**2) )
        arr = np.array([
            self.player.x, self.player.y,
            *self.player.vx, *self.player.vy,
            *[i.x-self.player.x for i in self.targets],
            *[i.y-self.player.y for i in self.targets],
        ])
        return arr









class GuidedNetwork:



    def __init__(self, layers):
        self.weights : List[np.ndarray] = []
        self.biases : List[np.ndarray] = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.normal(
                0,
                (2/layers[i])**0.5,
                (layers[i+1], layers[i])
            ))
        for i in range(len(layers)-1):
            self.biases.append(np.array([0.0]*layers[i+1]))

    def forward(self, layer):
        results = [layer.copy()]
        for i in range(len(self.weights)):
            layer = np.dot(self.weights[i], layer) + self.biases[i]
            if i+1!=len(self.weights):
                layer = np.maximum(0, layer)
            results.append(layer.copy())
        return results


    def backwardsOne(self, netState, diff, n_weights, n_biases):
        diff = np.array(diff)

        for lay in range(len(self.weights)-1, -1, -1):
            neurons = netState[lay]
            nextDiff = diff

            diff = np.dot(self.weights[lay].transpose(), diff)
            diff = np.where(neurons>0.001, diff, 0)
            n_weights[lay] += np.outer(nextDiff, neurons)
            n_biases[lay] += nextDiff


    def getCopiedShapes(self):
        return [np.zeros(i.shape) for i in self.weights], [np.zeros(i.shape) for i in self.biases]

    def applyWeights(self, nW, nB, learningRate, weightsDecay = 0.0001):
        weightsDecay = 1-weightsDecay
        for i in range(len(nW)):
            self.weights[i] = self.weights[i]*weightsDecay + nW[i]*learningRate
            self.biases[i] += nB[i]*learningRate


    def backwardsAll(self, samples, expected, nW, nB):
        results = []
        for samp in range(len(samples)):
            vec = self.forward(samples[samp])
            results.append(vec[-1])
            diff = (np.array(expected[samp])-vec[-1])
            self.backwardsOne(vec, diff, nW, nB)
        return results


    def getInfo(self):
        res = ""
        maxW = tuple(int(math.log(max(max(abs(i)) for i in j))) for j in self.weights)
        maxB = tuple(int(math.log(max(abs(j)))) for j in self.biases)
        res += (f"Max W (log): {' '.join(map(str, maxW))} Max B (log): {' '.join(map(str, maxB))} "
                f"")
        return res







def normal_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * np.pi))+0.0001) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2 + 0.0001))





class ReinforcedNetwork:


    def __init__(self, layers):
        self.weights : List[np.ndarray] = []
        self.biases : List[np.ndarray] = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.normal(
                0,
                (2/layers[i])**0.5,
                (layers[i+1], layers[i])
            ))
        for i in range(len(layers)-1):
            self.biases.append(np.array([0.0]*layers[i+1]))


    def forward(self, layer):
        results = [layer.copy()]
        for i in range(len(self.weights)):
            layer = np.dot(self.weights[i], layer) + self.biases[i]
            if i+1!=len(self.weights):
                layer = np.maximum(0, layer)
            results.append(layer.copy())

        #print()
        #print(list(int(max(i)) for i in results))
        #print(max(max(max(i) for i in j) for j in self.weights))
        #print(max(max(i) for i in self.biases))
        return results



    def backwardsOne(self, netState, diff, nWeights, nBiases, stepSize):
        diff = np.array(diff)

        for lay in range(len(self.weights)-1, -1, -1):
            neurons = netState[lay]
            #print(int(max(neurons)), int(max(diff)))
            #print(max(max(max(i) for i in j) for j in self.weights))
            nWeights[lay] += np.outer(diff, neurons)*stepSize
            nBiases[lay] += diff*stepSize
            diff = np.dot(self.weights[lay].transpose(), diff)
            diff = np.where(neurons>0.001, diff, 0)


    def getNeutralWeights(self):

        n_weights = [np.zeros(i.shape) for i in self.weights]
        n_biases = [np.zeros(i.shape) for i in self.biases]

        return n_weights, n_biases

    def getDimensions(self):
        return self.weights[0].shape[1], self.weights[-1].shape[0]
    def getCopiedWeights(self):
        return [i.copy() for i in self.weights], [i.copy() for i in self.biases]

    def applyWeights(self, n_weights, n_biases):
        self.weights = n_weights
        self.biases = n_biases

    def getInfo(self):
        res = ""
        maxW = tuple(int(math.log(max(max(abs(i)) for i in j))) for j in self.weights)
        maxB = tuple(int(math.log(max(abs(j)))) for j in self.biases)
        res += (f"Max W (log): {' '.join(map(str, maxW))} Max B (log): {' '.join(map(str, maxB))} "
                f"")
        return res


    def backwardsAll(self, actions, oldProbs, rewards, netStates, temp_weights, temp_biases, mxDev, learningRate):
        ln = len(actions)
        choices = len(netStates[0][-1])//2

        n_weights, n_biases = self.getNeutralWeights()



        for act in range(ln):
            correspondingReward = max(-40, min(40, rewards[act]))


            diff = []
            for choice in range(choices):


                action = actions[act][choice]
                expectedRaw = netStates[act][-1][choice*2]
                deviationRaw = netStates[act][-1][choice*2+1]

                expectedReal = math.tanh(expectedRaw)
                deviationReal = math.tanh(deviationRaw)/2+0.5


                oldActionProb = oldProbs[act][choice]
                currActionProb = normal_pdf(actions[act][choice], expectedReal, deviationReal)
                probsRatio = currActionProb/oldActionProb

                lossClipDeviation = correspondingReward*probsRatio
                if correspondingReward >= 0:
                    if probsRatio > 1+mxDev:
                        lossClipDeviation = 0
                if correspondingReward < 0:
                    if probsRatio < 1-mxDev:
                        lossClipDeviation = 0






                diffU = (action-expectedReal)/deviationReal**2
                diffU *= 1/math.cosh(expectedRaw)**2

                diffD = (action-expectedReal)**2/deviationReal**3-1/deviationReal
                diffD *= 1/math.cosh(deviationRaw)**2
                diffD *= (1/2)



                diffU *= lossClipDeviation
                diffD *= lossClipDeviation
                diff.append(diffU)
                diff.append(diffD)

            self.backwardsOne(netStates[act], diff,
                              nWeights=n_weights, nBiases=n_biases, stepSize=1/ln)

        lengW = 1
        lengW += sum(abs(j) for lay in n_weights for i in lay for j in i)
        lengW += sum(abs(i) for lay in n_biases for i in lay)

        if lengW<20:
            lengW = 5

        lengW /= 5

        for i in range(len(temp_weights)):
            temp_weights[i] += n_weights[i]*(learningRate/lengW)
            temp_biases[i] += n_biases[i]*(learningRate/lengW)





class ActorCritic:

    def __init__(self):
        self.actor : ReinforcedNetwork = ReinforcedNetwork(layers = [8+NUMBER_OF_TARGETS*2, 64, 64, 32, 4])
        self.critic : GuidedNetwork = GuidedNetwork(layers = [8+NUMBER_OF_TARGETS*2, 32, 32, 1])









if True:
    SCORE_TARGET = 20
    SCORE_TIME = 0.05
    SCORE_SPEED = 0.015
    SCORE_DIST = 0.001


def getReward(stage: Stage):
    score = 0

    score += stage.collide()*SCORE_TARGET
    stage.populate()
    score -= SCORE_TIME
    closestObst = stage.getClosestObst()
    v1 = (closestObst.x-stage.player.x, closestObst.y-stage.player.y)

    if True:
        vv1 = (stage.player.vx[0], stage.player.vy[0])
        lngth1 = (vv1[0]**2+vv1[1]**2)**0.5 + 0.001
        lngth2 = (v1[0]**2+v1[1]**2)**0.5 + 0.001
        scal = (v1[0]*vv1[0]+v1[1]*vv1[1])/lngth1/lngth2
        #print(scal)
        if True or (scal<0):
            score += SCORE_SPEED*scal*min(1, lngth1*4)

    if (abs(stage.player.x-0.5)>0.5 or abs(stage.player.y-0.5)>0.5):
        score = -100

    score += SCORE_DIST * (1/(v1[0]**2+v1[1]**2+0.5)**0.5-2/3)*3/4
    #score += 0.05 / (max(abs(stage.player.x-0.5), abs(stage.player.y-0.5))+1)


    return score


def evalOutput(layer):

    acts = []
    actions_probs = []
    for i in range(len(layer)//2):
        expectedRaw = layer[i*2]
        deviationRaw = layer[i*2+1]

        expectedReal = math.tanh(expectedRaw)
        deviationReal = math.tanh(deviationRaw)/2+0.5

        acts.append(np.random.normal(expectedReal, deviationReal))
        actions_probs.append(normal_pdf(acts[-1], expectedReal, deviationReal))
    return acts, actions_probs




def getSamples(forward, lifeT):
    stage = Stage()
    stage.player.radius *= 0.6
    stage.tick(1.0)
    actions = []
    actionsProbs = []
    rewards = []
    n_states = []
    env_states = []

    ticksUsed = 0
    sumReward = 0
    for tick in range(lifeT):
        inp = stage.getData()
        vec = forward(inp)

        acts, actions_probs = evalOutput(vec[-1])

        stage.move(tuple(i/6 for i in acts))
        stage.tick(0.25)

        score = getReward(stage)
        rewards.append(max(-50, min(50, score)))


        stage.populate()

        actions.append(acts)
        actionsProbs.append(actions_probs)
        n_states.append(vec)
        env_states.append(inp)


        if (score<-30):
            #print(f"iter {tick} finished early")
            break

    return actions, actionsProbs, rewards, n_states, env_states








def trainN(netAC, maxSteps, desiredAccuracy):
    episodesPerIter = 2
    ticksPerEpisode = 500
    rewardsDecay = 0.97
    learningRate = 0.1
    learningRateCritic = 0.25
    weightsDecay=0.0001
    steps_in_clipped = 2
    stepsCritic = 3
    clipped_deviation = 0.3


    weightsDecay = 1-weightsDecay
    bestWeights = netAC.actor.getCopiedWeights()
    mxVal = -1e9

    floatingAvQ = []
    floatingAv = 0
    itersUsed = 0
    globReward = 0
    fl = 0
    while True:
        if itersUsed >= maxSteps:
            break

        if floatingAv > 300 and fl==1:
            global SCORE_TIME, SCORE_SPEED, SCORE_DIST
            SCORE_TIME *= 2
            SCORE_SPEED /= 8
            SCORE_DIST /= 8
            mxVal = -1e9
            fl = 2
            print("\n"*10+"Difficulty increased")
        if (itersUsed>=maxSteps/10 and fl==0 and floatingAv>150):
            episodesPerIter = 15
            ticksPerEpisode = 500
            learningRate = 0.1
            #net.applyWeights(*bestWeights)
            floatingAvQ = [0]*10
            mxVal = -1e9
            fl=1
            print("\n"*10+"Reconfigured")


        iterReward = 0
        iterRewardExp = 0
        iterSteps = 0

        samples = []
        for i in range(episodesPerIter):
            #print(max(max(max(i) for i in j) for j in net.weights))
            samples.append(getSamples(netAC.actor.forward, ticksPerEpisode))
            n_rewards = samples[-1][2]
            iterSteps += len(n_rewards)
            itersUsed += len(n_rewards)*(1+steps_in_clipped+int(stepsCritic/1.5+0.5))


        advantages = []

        nW, nB = netAC.critic.getCopiedShapes()
        for samp in range(episodesPerIter):
            n_rewards = samples[samp][2]
            ln = len(n_rewards)
            correspondingRewards = [0]*(ln+1)
            for act in range(ln-1, -1, -1):
                correspondingRewards[act] = n_rewards[act]+correspondingRewards[act+1]*rewardsDecay
            correspondingRewards = np.array(correspondingRewards[:ln])

            iterReward += sum(correspondingRewards)/episodesPerIter
            globReward += sum(correspondingRewards)




            if True:
                env_states = samples[samp][4]
                for criticIter in range(stepsCritic):
                    expectedRewards = netAC.critic.backwardsAll(env_states, correspondingRewards, nW, nB)
                    expectedRewards = np.array([i[0] for i in expectedRewards])
                    if criticIter==0:
                        iterRewardExp += sum(expectedRewards)/episodesPerIter
                        correspondingRewards -= expectedRewards
            advantages.append(correspondingRewards)

        netAC.critic.applyWeights(nW, nB, learningRate=learningRateCritic/episodesPerIter/ticksPerEpisode)


        floatingAvQ.append(iterReward)
        if len(floatingAvQ)>20:
            floatingAvQ.pop(0)
        floatingAv = sum(floatingAvQ)/len(floatingAvQ)



        allRewards = np.array([j for i in advantages for j in i])
        meanReward = allRewards.mean()
        stdReward = allRewards.std()

        for samp in range(episodesPerIter):
            advantages[samp] = (np.array(advantages[samp])-meanReward)/(stdReward+0.1)

        if True:
            for clip_step in range(steps_in_clipped):

                nW, nB = netAC.actor.getCopiedWeights()
                if (clip_step==0):
                    for i in range(len(nW)):
                        nW[i] *= weightsDecay

                for samp in range(episodesPerIter):
                    old_acts, old_probs, old_rewards, old_states, old_env_states = samples[samp]
                    n_states = list(netAC.actor.forward(i) for i in old_env_states)
                    netAC.actor.backwardsAll(old_acts, old_probs, advantages[samp], n_states, nW, nB, clipped_deviation, learningRate=learningRate/episodesPerIter/steps_in_clipped)
                netAC.actor.applyWeights(nW, nB)


        if iterSteps*2 > ticksPerEpisode*episodesPerIter:
            if iterReward > mxVal*0.6:
                mxVal = max(iterReward, mxVal)
                bestWeights = netAC.actor.getCopiedWeights()


        if iterSteps*2 > ticksPerEpisode*episodesPerIter:
            print(f"{itersUsed}/{maxSteps} | Iteration steady ({iterSteps}), reward: {iterReward} (av: {floatingAv}, exp: {iterRewardExp})")
            print(netAC.actor.getInfo())
        if floatingAv > desiredAccuracy:
            break


    netAC.actor.applyWeights(*bestWeights)

    return globReward/maxSteps















if True:

    SEED = 150
    random.seed(SEED)
    np.random.seed(SEED)

    nett = ActorCritic()

    tm = time.time()
    result = 0
    print("training started")
    result = trainN(nett, 1000000, desiredAccuracy = 50000)
    print(f"average reward = {result}")
    tm1 = time.time()
    print(f"training took {tm1-tm} seconds")

    #exit()

    stage = Stage()
    stage.populate()
    Controller(stage, nett.actor).run()


























