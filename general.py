import math
from random import Random

import numpy as np






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


    def __init__(self, seed):
        self.random = Random(seed).random
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
            self.targets[-1].x = self.random()
            self.targets[-1].y = self.random()

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













def saveWeights(file, net):
    data = {}
    for i, w in enumerate(net.actor.weights):
        data[f"actor_w_{i}"] = w
    for i, b in enumerate(net.actor.biases):
        data[f"actor_b_{i}"] = b
    for i, w in enumerate(net.critic.weights):
        data[f"critic_w_{i}"] = w
    for i, b in enumerate(net.critic.biases):
        data[f"critic_b_{i}"] = b

    if isinstance(file, (str, bytes)) or hasattr(file, "__fspath__"):
        with open(file, "wb") as f:
            np.savez(f, **data)
    else:
        np.savez(file, **data)

def loadWeights(file, net):
    def _read_arrays(f):
        archive = np.load(f)
        try:
            n_actor = 0
            while f"actor_w_{n_actor}" in archive.files:
                n_actor += 1
            n_critic = 0
            while f"critic_w_{n_critic}" in archive.files:
                n_critic += 1

            if n_actor == 0 or n_critic == 0:
                raise ValueError("File does not contain saved ActorCritic weights")

            actor_w = [np.array(archive[f"actor_w_{i}"], copy=True) for i in range(n_actor)]
            actor_b = [np.array(archive[f"actor_b_{i}"], copy=True) for i in range(n_actor)]
            critic_w = [np.array(archive[f"critic_w_{i}"], copy=True) for i in range(n_critic)]
            critic_b = [np.array(archive[f"critic_b_{i}"], copy=True) for i in range(n_critic)]
        finally:
            archive.close()
        return actor_w, actor_b, critic_w, critic_b

    if isinstance(file, (str, bytes)) or hasattr(file, "__fspath__"):
        with open(file, "rb") as f:
            actor_w, actor_b, critic_w, critic_b = _read_arrays(f)
    else:
        actor_w, actor_b, critic_w, critic_b = _read_arrays(file)

    actor_layers = [actor_w[0].shape[1]] + [w.shape[0] for w in actor_w]
    critic_layers = [critic_w[0].shape[1]] + [w.shape[0] for w in critic_w]

    net.actor.weights = actor_w
    net.actor.biases = actor_b
    net.critic.weights = critic_w
    net.critic.biases = critic_b

    return net

def weightsPath(inp_outp_size):
    name = f"stage_weights_{inp_outp_size[0]}-{inp_outp_size[-1]}.npz"
    return name
