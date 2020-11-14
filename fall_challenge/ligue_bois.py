import os
import sys
import time
from dataclasses import dataclass, replace
from enum import Enum

import math
import numpy as np
from typing import List, Tuple

NB_GEMMES = 4
MAX_NB_ITEMS = 10
SEUIL_TIME = 0.042
NB_LEARN_MAX = 10
NB_SORTS_INITIAUX = 5  # Avec REST
DECROISSANCE_SORTS = 1.2
NB_POTIONS_CRAFTABLE_MAX = 6
COEF_COUT_VS_REWARD = 2


def debug(message: str, end="\n"):
    print(message, file=sys.stderr, flush=True, end=end)
    # pass


def read_input():
    return input()


class ActionType(Enum):
    SORT = "CAST"
    SORT_OPPONENT = "OPPONENT_CAST"
    POTION = "BREW"
    REST = "REST"
    LEARN = "LEARN"
    WAIT = "WAIT"


class Inventory:
    def __init__(self, inventory: np.array):
        self.inv = np.array(inventory)

    def update_inventory_with_sort(self, sort: 'Sort'):
        if sort == ActionType.REST.value:
            return
        self.inv -= sort.cout
        for i in range(NB_GEMMES):
            sum = self.get_nb_items()
            max_to_add = 10 - sum
            self.inv[i] += min(max_to_add, sort.reward[i])

    def update_inventory_with_learn(self, learn: 'Learn'):
        self.inv -= learn.achat_cout
        max_gainable = MAX_NB_ITEMS - self.get_nb_items()
        self.inv += min(max_gainable, learn.tax_count)

    def update_inventory_with_sorts(self, sorts: List['Sort']):
        for sort in sorts:
            self.update_inventory_with_sort(sort)

    def get_nb_items(self) -> int:
        return np.sum(self.inv)

    def is_full(self) -> bool:
        return self.get_nb_items() >= MAX_NB_ITEMS

    def copy(self):
        return Inventory(self.inv)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.inv}"


@dataclass
class Sort:
    def __init__(self,
                 id: int,
                 type: str,
                 delta_0: int,
                 delta_1: int,
                 delta_2: int,
                 delta_3: int,
                 price: int,
                 tome_index: int,
                 tax_count: int,
                 castable: bool,
                 repeatable: bool):
        self.id = id
        self.type = type
        self.price = price
        self.castable = castable
        self.tome_index = tome_index
        self.tax_count = tax_count
        self.repeatable = repeatable

        self.cout = np.array([delta_0, delta_1, delta_2, delta_3])
        self.reward = self.cout.copy()
        self.reward[self.reward < 0] = 0
        self.cout[self.cout > 0] = 0
        self.cout = - self.cout

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Sort({str(self.reward - self.cout)}({'O' if self.castable else 'X'}))"

    def is_castable(self, inventory: Inventory) -> bool:
        return self.castable \
               and all(inventory.inv - self.cout >= 0) \
               and np.sum(inventory.inv - self.cout + self.reward) <= MAX_NB_ITEMS

    def get_used(self):
        used_sort = Sort(self.id, self.type, 0, 0, 0, 0, self.price, self.tome_index, self.tax_count, False,
                         self.repeatable)
        used_sort.cout = self.cout
        used_sort.reward = self.reward
        return used_sort

    def copy(self) -> 'Sort':
        sort_copied = Sort(self.id, self.type, 0, 0, 0, 0, self.price, self.tome_index, self.tax_count, self.castable,
                           self.repeatable)
        sort_copied.cout = self.cout
        sort_copied.reward = self.reward
        return sort_copied

    def rest(self) -> 'Sort':
        self.castable = True
        return self

    def get_estimated_gain(self, sort_indice: int, nb_potions_a_faire: int) -> float:
        gain = np.sum((self.reward - self.cout * COEF_COUT_VS_REWARD) * np.array([1, 2, 3, 4]))
        return gain
        # if sort_indice > NB_POTIONS_CRAFTABLE_MAX:
        #     return 0
        # gain = np.sum((self.reward - self.cout) * np.array([1, 2, 3, 4]))
        # if sort_indice >= NB_SORTS_INITIAUX:
        #     denom = math.pow(sort_indice + 1 - NB_SORTS_INITIAUX, DECROISSANCE_SORTS)
        #     num = (nb_potions_a_faire / NB_POTIONS_CRAFTABLE_MAX)
        #     coef = num / denom
        #     # debug(f"indice = {sort_indice} nb_potions_a_faire = {nb_potions_a_faire} num = {num} denom = {denom} coef = {coef}")
        #     gain *= coef
        # return gain


class Learn(Sort):
    def __init__(self,
                 id: int,
                 type: str,
                 delta_0: int,
                 delta_1: int,
                 delta_2: int,
                 delta_3: int,
                 price: int,
                 tome_index: int,
                 tax_count: int,
                 castable: bool,
                 repeatable: bool):
        super().__init__(id, type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable,
                         repeatable)
        self.achat_cout = np.array([tome_index, 0, 0, 0])

    def is_buyable(self, inventory: Inventory) -> bool:
        return inventory.inv[0] >= self.achat_cout[0]

    def get_sort(self) -> Sort:
        sort = Sort(self.id, ActionType.SORT.value, 0, 0, 0, 0, 0, self.tome_index, self.tax_count, True,
                    self.repeatable)
        sort.cout = self.cout
        sort.reward = self.reward
        return sort


def update_sorts_with_sort(sorts: List[Sort], sort: Sort) -> List[Sort]:
    new_sorts = [s.copy() if not isinstance(s, str) else s for s in sorts]
    if sort == ActionType.REST.value:
        new_sorts = [s.rest() if not isinstance(s, str) else s for s in new_sorts]
    else:
        new_sorts = [(s if s.id != sort.id else s.get_used()) if not isinstance(s, str) else s for s in new_sorts]
    return new_sorts


def update_sorts_with_learn(sorts: List[Sort], learn: Learn) -> List[Sort]:
    new_sorts = sorts
    new_sorts.append(learn.get_sort())
    return new_sorts


def update_learns_with_learn(learns: List[Learn], learn: Learn) -> List[Learn]:
    new_learns = [l for l in learns if l.id != learn.id]
    return new_learns


class Action:
    @staticmethod
    def read():
        inp = read_input()
        debug(f"{inp}")
        action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable, repeatable \
            = inp.split()
        action_id = int(action_id)
        delta_0 = int(delta_0)
        delta_1 = int(delta_1)
        delta_2 = int(delta_2)
        delta_3 = int(delta_3)
        price = int(price)
        tome_index = int(tome_index)
        tax_count = int(tax_count)
        castable = castable != "0"
        repeatable = repeatable != "0"
        if action_type == ActionType.POTION.value:
            return Potion(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                          castable, repeatable)
        elif action_type == ActionType.SORT.value:
            return Sort(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                        castable, repeatable)
        elif action_type == ActionType.SORT_OPPONENT.value:
            return Sort(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                        castable, repeatable)
        elif action_type == ActionType.LEARN.value:
            return Learn(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                         castable, repeatable)


class Potion:
    def __init__(self,
                 id: int,
                 type: str,
                 delta_0: int,
                 delta_1: int,
                 delta_2: int,
                 delta_3: int,
                 price: int,
                 tome_index: int,
                 tax_count: int,
                 castable: bool,
                 repeatable: bool):
        self.id = id
        self.type = type
        self.cout = np.array([delta_0, delta_1, delta_2, delta_3])
        self.cout = - self.cout
        self.price = price
        self.castable = castable
        self.tome_index = tome_index
        self.tax_count = tax_count
        self.repeatable = repeatable
        self.path = None

    @staticmethod
    def read():
        action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable, repeatable \
            = read_input().split()
        action_id = int(action_id)
        delta_0 = int(delta_0)
        delta_1 = int(delta_1)
        delta_2 = int(delta_2)
        delta_3 = int(delta_3)
        price = int(price)
        tome_index = int(tome_index)
        tax_count = int(tax_count)
        castable = castable != "0"
        repeatable = repeatable != "0"
        return Potion(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                      castable, repeatable)

    def get_score(self, m: 'Model') -> float:
        difficulty = self.difficulty_to_get(m)
        if difficulty is None:
            return -1
        return self.get_price() / (difficulty + 1)

    def difficulty_to_get(self, m: 'Model') -> int:
        if not self.path:
            self.compute_path(m)
        if self.path is None:
            return None
        return len(self.path)

    def compute_path(self, m: 'Model'):
        end = Node(Inventory(self.cout), [], learns=None, precedent=None, goal=None, sort_used=None, m=m)
        start = Node(m.me.inventory, m.sorts + [ActionType.REST.value], learns=m.learns, precedent=None, goal=end,
                     sort_used=None, m=m)
        self.path = a_star(start, end, m)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Potion({str(self.cout)}({self.get_price()}))"

    def get_price(self) -> float:
        return self.price

    def distance_to_inventory(self, inventory: Inventory):
        return dist_a_to_b(inventory.inv, self.cout)


class Player:
    def __init__(self, inv0: int, inv1: int, inv2: int, inv3: int, score: int):
        self.inventory = Inventory(np.array([inv0, inv1, inv2, inv3]))
        self.score = score

    def can_buy(self, potion: Potion) -> bool:
        return np.all(self.inventory.inv - potion.cout)

    def buy(self, potion: Potion) -> 'Player':
        new_player = Player(self.inventory.inv[0], self.inventory.inv[1], self.inventory.inv[2], self.inventory.inv[3],
                            self.score)
        new_player.inventory -= potion.cout
        new_player.score += potion.get_price()
        return new_player

    @staticmethod
    def read():
        inp = read_input()
        debug(f"{inp}")
        return Player(*[int(i) for i in inp.split()])


def compute_heuristique_sorts(sorts: List[Sort], m: 'Model'):
    total = 0
    for i, sort in enumerate(sorts):
        if not isinstance(sort, str):
            total += sort.get_estimated_gain(i, m.nb_potions_to_craft)
    return total


def dist_a_to_b(a: np.array, b: np.array) -> float:
    a = a.copy()
    b = b.copy()
    total = 0
    for gemme_indice in reversed(range(NB_GEMMES)):
        while a[gemme_indice] > 0 and sum(b[gemme_indice:]) > 0:
            a[gemme_indice] -= 1
            ind_b = np.argmax(b[gemme_indice:] > 0) + gemme_indice
            b[ind_b] -= 1
            total += gemme_indice + 1
    return total


def compute_heuristique(n: 'Node', goal: 'Node', m: 'Model') -> float:
    total = dist_a_to_b(n.inventory.inv, goal.inventory.inv)

    from_sorts = compute_heuristique_sorts(n.sorts, m)
    # debug(f"heuristique from gemmes = {total} from sorts = {from_sorts}({len(n.sorts)})")
    total += from_sorts
    sorts_castables = [s for s in n.sorts if isinstance(s, str) or s.castable]
    total += (0.01 * len(sorts_castables))
    return total


def is_rest_castable(sorts: List[Sort]) -> bool:
    sorts_without_rest = [s for s in sorts if not isinstance(s, str)]
    return not all([s.castable for s in sorts_without_rest])


class Node:
    def __init__(self,
                 inventory: Inventory,
                 sorts: List[Sort],
                 learns: List[Learn],
                 precedent: 'Node',
                 goal: 'Node',
                 sort_used: Sort,
                 m: 'Model'):
        self.inventory = inventory
        self.sorts = sorts
        self.learns = learns
        if precedent:
            self.cout = precedent.cout + 1
        else:
            self.cout = 0
        self.precedent = precedent
        if goal:
            self.heuristique = compute_heuristique(self, goal, m)
            self.value = self.cout + self.heuristique
            self.sort_used = sort_used
        else:
            self.heuristique = None
            self.value = None
            self.sort_used = None

    def get_voisins(self, goal: 'Node', m: 'Model') -> List['Node']:
        nodes = []
        for sort in self.get_possible_sorts():
            new_inventory = self.inventory.copy()
            new_inventory.update_inventory_with_sort(sort)
            new_sorts = update_sorts_with_sort(self.sorts, sort)
            node = Node(new_inventory, new_sorts, self.learns, self, goal, sort, m)
            nodes.append(node)

        for learn in self.get_possible_learns():
            if learn.get_estimated_gain(0, 0) >= 2:
                new_inventory = self.inventory.copy()
                new_inventory.update_inventory_with_learn(learn)
                new_sorts = update_sorts_with_learn(self.sorts, learn)
                new_learns = update_learns_with_learn(self.learns, learn)
                node = Node(new_inventory, new_sorts, new_learns, self, goal, learn, m)
                nodes.append(node)

        return nodes

    def get_possible_sorts(self) -> List[Sort]:
        sort_castables = []
        for s in self.sorts:
            if s == ActionType.REST.value:
                if is_rest_castable(self.sorts):
                    sort_castables.append(ActionType.REST.value)
            elif s.is_castable(self.inventory):
                sort_castables.append(s)
        return sort_castables

    def get_possible_learns(self) -> List[Learn]:
        learns_buyables = []
        for l in self.learns:
            if l.is_buyable(self.inventory):
                learns_buyables.append(l)
        return learns_buyables

    def is_better_than(self, other: 'Node') -> bool:
        return all(self.inventory.inv - other.inventory.inv >= 0)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.inventory}+{self.sorts}({self.value})"


def insert_at_good_place(opened: List[Node], voisin: Node) -> List[Node]:
    for i, o in enumerate(opened):
        if o.value >= voisin.value:
            opened.insert(i, voisin)
            return opened
    opened.append(voisin)
    return opened


def insert_in_opened(opened: List[Node], voisin: Node) -> List[Node]:
    is_in_opened = False
    for i, o in enumerate(opened):
        if o == voisin:
            if o.value < voisin.value:
                opened[i] = voisin
                is_in_opened = True
            else:
                break
    if not is_in_opened:
        opened = insert_at_good_place(opened, voisin)
    return opened


def compute_path_backward(current: Node) -> List[Sort]:
    path = []
    while current.precedent:
        path.append(current.sort_used)
        current = current.precedent
    return path


def a_star(start: Node, end: Node, m: 'Model') -> List[Sort]:
    opened = [start]
    closed = []

    while len(opened) > 0:
        if time.time() - m.debut_time >= SEUIL_TIME:
            break

        current = opened[-1]
        opened = opened[:-1]

        if current.is_better_than(end):
            path = compute_path_backward(current)
            return path

        voisins = current.get_voisins(end, m)

        for voisin in voisins:
            if voisin not in closed:
                opened = insert_in_opened(opened, voisin)

        closed.append(current)

    debug(f"Impossible de trouver un chemin de {start.inventory.inv} vers {end.inventory.inv} !")
    return None


class Model:
    def __init__(self, potions: List[Potion], sorts: List[Sort], learns: List[Learn], opp_sorts: List[Sort], me: Player,
                 opp: Player, nb_potions_to_craft: int):
        self.potions = potions
        self.sorts = sorts
        self.learns = learns
        self.opp_sorts = opp_sorts
        self.me = me
        self.opp = opp
        self.nb_potions_to_craft = nb_potions_to_craft
        self.debut_time = 0

    @staticmethod
    def read(nb_potions_to_craft: int):
        nb_objects = int(read_input())
        debug(f"{nb_objects}")
        actions = [Action.read() for _ in range(nb_objects)]
        potions = [a for a in actions if a.type == ActionType.POTION.value]
        sorts = [a for a in actions if a.type == ActionType.SORT.value]
        learns = [a for a in actions if a.type == ActionType.LEARN.value]
        opp_sorts = [a for a in actions if a.type == ActionType.SORT_OPPONENT.value]
        me = Player.read()
        opp = Player.read()
        return Model(potions, sorts, learns, opp_sorts, me, opp, nb_potions_to_craft)


def find_greedy_objectif(m: Model) -> Potion:
    sorted(m.potions, key=lambda p: p.distance_to_inventory(m.me.inventory))
    for potion in m.potions:
        potion.compute_path(m)
    potion_scores = [p.get_score(m) for p in m.potions]
    best_potion = m.potions[np.argmax([potion_scores])]
    return best_potion


def apply_simple_algorithm(m: 'Model'):
    for sort in m.sorts:
        if sort.is_castable(m.me.inventory):
            print(f"{ActionType.SORT.value} {sort.id} DEFAUT SORT {sort}")
            return
    print(f"{ActionType.REST.value} DEFAUT REST !")
    return


def run():
    current_round = 0
    nb_potions_to_craft = NB_POTIONS_CRAFTABLE_MAX
    while True:
        m = Model.read(nb_potions_to_craft)

        m.debut_time = time.time()
        potion_objectif = find_greedy_objectif(m)
        print(f"round = {current_round} (time={(time.time() - m.debut_time) * 1000})", file=sys.stderr, flush=True)

        if potion_objectif.path is not None:
            print(f"potion_objectif = (id={potion_objectif.id}, len={len(potion_objectif.path)}, score={potion_objectif.get_score(m)})", file=sys.stderr, flush=True)
            if potion_objectif.path:
                if potion_objectif.get_score(m) == -1:
                    print(f"{ActionType.WAIT.value} ERROR")
                action = potion_objectif.path[-1]
                if action == ActionType.REST.value:
                    print(f"{ActionType.REST.value} REST {potion_objectif}")
                elif action.type == ActionType.SORT.value:
                    print(f"{ActionType.SORT.value} {action.id} SORT {potion_objectif}")
                elif action.type == ActionType.LEARN.value:
                    print(f"{ActionType.LEARN.value} {action.id} LEARN ! <3")
                potion_objectif.path = potion_objectif.path[:-1]
            else:
                print(f"{ActionType.POTION.value} {potion_objectif.id} {potion_objectif} !!! :D")
                nb_potions_to_craft -= 1
        else:
            print(f"potion_objectif = (id={potion_objectif.id}, len={None}, score={potion_objectif.get_score(m)})", file=sys.stderr, flush=True)
            apply_simple_algorithm(m)


        current_round += 1


if __name__ == '__main__':
    if os.path.exists('in.txt'):
        sys.stdin = open('in.txt', 'r', newline='\n')
    run()
