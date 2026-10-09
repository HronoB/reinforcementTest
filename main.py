import math
import random
import time

import numpy as np

from stageController import Controller


class Stage:

    class FlatObject:
        def __init__(self):
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
                self.vx[i] /= math.exp(delta)
                self.vy[i] /= math.exp(delta)

        def accel(self, order, vx, vy):
            self.vx[order] += vx
            self.vy[order] += vy


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
        while len(self.targets)<2:
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
            *[i.x for i in self.targets],
            *[i.y for i in self.targets],
        ])
        return arr











class ReinforcedNetwork:


    def __init__(self, layers = [12, 64, 64, 32, 4]):
        self.weights = []
        self.biases = []
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

    def getCopiedWeights(self):
        return [i.copy() for i in self.weights], [i.copy() for i in self.biases]

    def applyWeights(self, n_weights, n_biases):
        self.weights = n_weights
        self.biases = n_biases



    def backwardsAll(self, actions, rewards, netStates, temp_weights, temp_biases, learningRate):
        ln = len(actions)
        choices = len(netStates[0][-1])//2

        weightsDecay=0.0001

        n_weights, n_biases = self.getNeutralWeights()


        for act in range(ln):
            correspondingReward = max(-50, min(60, rewards[act]))
            diff = []
            for choice in range(choices):
                action = actions[act][choice]
                expectedRaw = netStates[act][-1][choice*2]
                deviationRaw = netStates[act][-1][choice*2+1]

                expectedReal = math.tanh(expectedRaw)
                deviationReal = math.tanh(deviationRaw)/2+0.5



                diffU = (action-expectedReal)/deviationReal**2
                diffU *= 2/math.cosh(expectedReal)**2

                diffD = (action-expectedReal)**2/deviationReal**2-1
                diffD *= (1/2) * 2/math.cosh(expectedReal)**2



                diffU *= correspondingReward
                diffD *= correspondingReward
                diff.append(diffU)
                diff.append(diffD)

            self.backwardsOne(netStates[act], diff,
                              nWeights=n_weights, nBiases=n_biases, stepSize=1/ln)

        lengW = sum(abs(j) for lay in n_weights for i in lay for j in i)+1
        lengB = sum(abs(i) for lay in n_biases for i in lay)+1

        if lengW<100:
            lengW = 5
        if lengB<100:
            lengB = 5

        lengW /= 5
        lengB /= 5

        for i in range(len(temp_weights)):
            temp_weights[i] += -self.weights[i]*weightsDecay
            temp_weights[i] += n_weights[i]*(learningRate/lengW)
            temp_biases[i] += n_biases[i]*(learningRate/lengB)



def getSamples(net, lifeT):
    stage = Stage()
    stage.tick(1.0)
    actions = []
    rewards = []
    nstates = []

    ticksUsed = 0
    sumReward = 0
    for tick in range(lifeT):
        inp = stage.getData()
        vec = net.forward(inp)
        acts = []
        for i in range(len(vec[-1])//2):
            expected = vec[-1][i*2]
            deviation = vec[-1][i*2+1]
            expected = math.tanh(expected)
            deviation = math.tanh(deviation)/2+0.5

            acts.append(np.random.normal(expected, deviation))
        score = 0
        stage.move(tuple(i/10 for i in acts))


        stage.tick(0.25)
        score += stage.collide()*20
        stage.populate()
        score -= 0.1
        closestObst = stage.getClosestObst()
        v1 = (closestObst.x-stage.player.x, closestObst.y-stage.player.y)

        if True:
            vv1 = (stage.player.vx[0], stage.player.vy[0])
            lngth1 = (vv1[0]**2+vv1[1]**2)**0.5 + 0.001
            lngth2 = (v1[0]**2+v1[1]**2)**0.5 + 0.001
            scal = (v1[0]*vv1[0]+v1[1]*vv1[1])/lngth1/lngth2
            #print(scal)
            if (scal>0):
                score += 0.1*scal*min(0.2, lngth1)

        if (abs(stage.player.x-0.5)>0.5 or abs(stage.player.y-0.5)>0.5):
            score = -100

        score += 0.05 / (v1[0]**2+v1[1]**2+0.1)**0.5
        #score += 0.05 / (max(abs(stage.player.x-0.5), abs(stage.player.y-0.5))+1)

        actions.append(acts)
        rewards.append(max(-50, min(50, score)))
        nstates.append(vec)


        ticksUsed = tick
        sumReward += rewards[-1]
        if (score<-30):
            #print(f"iter {tick} finished early")
            break

    return actions, rewards, nstates, ticksUsed+1, sumReward








def trainN(net, maxSteps, desiredAccuracy, learningRate):
    episodesPerIter = 4
    ticksPerEpisode = 800
    rewardsDecay = 0.98

    itersLeft = maxSteps
    globReward = 0
    while True:
        if itersLeft <= 0:
            break
        iterReward = 0
        iterSteps = 0
        nW, nB = net.getCopiedWeights()
        samples = []
        for i in range(episodesPerIter):
            #print(max(max(max(i) for i in j) for j in net.weights))
            samples.append(getSamples(net, ticksPerEpisode))
            n_acts, n_rewards, n_states, ticksUsed, sumReward = samples[-1]
            iterReward += sumReward/episodesPerIter
            iterSteps += ticksUsed
            globReward += sumReward
            itersLeft -= ticksUsed

        for samp in range(episodesPerIter):
            n_acts, n_rewards, n_states, ticksUsed, sumReward = samples[samp]
            ln = len(n_acts)
            correspondingRewards = np.zeros(ln+1)
            for act in range(ln-1, -1, -1):
                correspondingRewards[act] = n_rewards[act]+correspondingRewards[act+1]*rewardsDecay
            n_rewards[:] = correspondingRewards
            #print(type(samples[samp][1]))

        allRewards = np.array([j for i in samples for j in i[1]])
        meanReward = allRewards.mean()
        stdReward = allRewards.std()

        for samp in range(episodesPerIter):
            samples[samp][1][:] = (np.array(samples[samp][1])-meanReward)/(stdReward+0.001)

        for samp in range(episodesPerIter):
            n_acts, n_rewards, n_states, ticksUsed, sumReward = samples[samp]
            net.backwardsAll(n_acts, n_rewards, n_states, nW, nB, learningRate/episodesPerIter)


        if iterSteps*1.5 > ticksPerEpisode*episodesPerIter:
            print(f"Iteration steady ({iterSteps}), reward: {iterReward}")
        if iterReward > desiredAccuracy:
            break
        net.applyWeights(nW, nB)
    return globReward/maxSteps















if True:

    net = ReinforcedNetwork()

    tm = time.time()
    result = trainN(net, 5000000, desiredAccuracy = 500, learningRate = 0.02)
    print(f"average reward = {result}")
    tm1 = time.time()
    print(f"training took {tm1-tm} seconds")

    #exit()

    stage = Stage()
    stage.populate()
    Controller(stage, net).run()










