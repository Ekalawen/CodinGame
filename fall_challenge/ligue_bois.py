import sys
import time
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from typing import List, Tuple

READ_STDIN = True
# READ_STDIN = False

NB_GEMMES = 4
MAX_NB_ITEMS = 10


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


def update_sorts_with_sort(sorts: List[Sort], sort: Sort) -> List[Sort]:
    new_sorts = [s.copy() if not isinstance(s, str) else s for s in sorts]
    if sort == ActionType.REST.value:
        new_sorts = [s.rest() if not isinstance(s, str) else s for s in new_sorts]
    else:
        new_sorts = [(s if s.id != sort.id else s.get_used()) if not isinstance(s, str) else s for s in new_sorts]
    return new_sorts


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
        if action_type == "BREW":
            return Potion(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                          castable, repeatable)
        elif action_type == "CAST":
            return Sort(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
                        castable, repeatable)
        elif action_type == "OPPONENT_CAST":
            return Sort(action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count,
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
        return self.price / (difficulty + 1)

    def difficulty_to_get(self, m: 'Model') -> int:
        if not self.path:
            self.compute_path(m)
        if self.path is None:
            return None
        return len(self.path)

    def compute_path(self, m: 'Model'):
        end = Node(Inventory(self.cout), [], precedent=None, goal=None, sort_used=None)
        start = Node(m.me.inventory, m.sorts + [ActionType.REST.value], precedent=None, goal=end, sort_used=None)
        self.path = a_star(start, end)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Potion({str(self.cout)}({self.price}))"


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
        new_player.score += potion.price
        return new_player

    @staticmethod
    def read():
        inp = read_input()
        debug(f"{inp}")
        return Player(*[int(i) for i in inp.split()])


def best_fit(potions: List[Potion], me: Player, quantity: int) -> Tuple[List[Potion], Player]:
    if quantity == 1:
        best_potion = potions[np.array([c.price for c in potions if me.can_buy(c)]).argmax()]
        new_player = me.buy(best_potion)
        return [best_potion], new_player

    best_potions, best_player = None, None
    for potion in potions:
        new_player = me.buy(potion)
        new_potions = potions
        new_potions.remove(potion)
        res_potions, res_player = best_fit(new_potions, new_player, quantity - 1)
        res_potions.append(potion)
        if not best_player or res_player.score > best_player.score:
            best_potions = res_potions
            best_player = res_player
    return best_potions, best_player


def compute_best(potions: List[Potion], me: Player, opp: Player, nb_rounds_restant: int) -> Potion:
    best_potions, best_player = best_fit(potions, me, nb_rounds_restant)
    return best_potions[0]


def compute_heuristique(n: 'Node', goal: 'Node') -> float:
    total = 0
    start = n.inventory.inv.copy()
    to_do = goal.inventory.inv.copy()
    for gemme_indice in reversed(range(NB_GEMMES)):
        while start[gemme_indice] > 0 and sum(to_do[gemme_indice:]) > 0:
            start[gemme_indice] -= 1
            ind_to_do = np.argmax(to_do[gemme_indice:] > 0) + gemme_indice
            to_do[ind_to_do] -= 1
            total += gemme_indice + 1

    total += 0.1 * len(n.sorts)
    return total


def is_rest_castable(sorts: List[Sort]) -> bool:
    sorts_without_rest = [s for s in sorts if not isinstance(s, str)]
    return not all([s.castable for s in sorts_without_rest])


class Node:
    def __init__(self,
                 inventory: Inventory,
                 sorts: List[Sort],
                 precedent: 'Node',
                 goal: 'Node',
                 sort_used: Sort):
        self.inventory = inventory
        self.sorts = sorts
        if precedent:
            self.cout = precedent.cout + 1
        else:
            self.cout = 0
        self.precedent = precedent
        if goal:
            self.heuristique = compute_heuristique(self, goal)
            self.value = self.cout + self.heuristique
            self.sort_used = sort_used
        else:
            self.heuristique = None
            self.value = None
            self.sort_used = None

    def get_voisins(self, goal: 'Node') -> List['Node']:
        nodes = []
        for sort in self.get_possible_sorts():
            new_inventory = self.inventory.copy()
            new_inventory.update_inventory_with_sort(sort)
            new_sorts = update_sorts_with_sort(self.sorts, sort)
            node = Node(new_inventory, new_sorts, self, goal, sort)
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


def a_star(start: Node, end: Node) -> List[Sort]:
    opened = [start]
    closed = []

    while len(opened) > 0:
        current = opened[-1]
        opened = opened[:-1]

        if current.is_better_than(end):
            path = compute_path_backward(current)
            return path

        voisins = current.get_voisins(end)

        for voisin in voisins:
            if voisin not in closed:
                opened = insert_in_opened(opened, voisin)

        closed.append(current)

    debug(f"Impossible de trouver un chemin de {start.inventory.inv} vers {end.inventory.inv} !")
    return None


class Model:
    def __init__(self, potions: List[Potion], sorts: List[Sort], opp_sorts: List[Sort], me: Player, opp: Player):
        self.potions = potions
        self.sorts = sorts
        self.opp_sorts = opp_sorts
        self.me = me
        self.opp = opp

    @staticmethod
    def read():
        nb_objects = int(read_input())
        debug(f"{nb_objects}")
        actions = [Action.read() for _ in range(nb_objects)]
        potions = [a for a in actions if a.type == ActionType.POTION.value]
        sorts = [a for a in actions if a.type == ActionType.SORT.value]
        opp_sorts = [a for a in actions if a.type == ActionType.SORT_OPPONENT.value]
        me = Player.read()
        opp = Player.read()
        return Model(potions, sorts, opp_sorts, me, opp)


def find_greedy_objectif(m: Model) -> Potion:
    for potion in m.potions:
        potion.compute_path(m)
    potion_scores = [p.get_score(m) for p in m.potions]
    best_potion = m.potions[np.argmax([potion_scores])]
    return best_potion


def run():
    current_round = 0
    nb_potions_to_craft = 3
    while True:
        m = Model.read()

        debut = time.time()
        potion_objectif = find_greedy_objectif(m)
        print(f"round = {current_round} (time={(time.time() - debut)*1000})", file=sys.stderr, flush=True)

        print(f"potion_objectif = ({potion_objectif.id})={potion_objectif.path}", file=sys.stderr, flush=True)

        if potion_objectif.path:
            if potion_objectif.path[-1] == ActionType.REST.value:
                print(f"{ActionType.REST.value}")
            else:
                print(f"{ActionType.SORT.value} {potion_objectif.path[-1].id}")
            potion_objectif.path = potion_objectif.path[:-1]
        else:
            print(f"{ActionType.POTION.value} {potion_objectif.id}")

        current_round += 1


if __name__ == '__main__':
    if not READ_STDIN:
        sys.stdin = open('in.txt', 'r', newline='\n')
    run()
