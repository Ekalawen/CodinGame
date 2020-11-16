import os
import sys
import time
from enum import Enum

import numpy as np
from typing import List, Any

NB_GEMMES = 4
MAX_NB_ITEMS = 10
SEUIL_TIME = 0.042
NB_SORTS_INITIAUX = 5  # Avec REST
DECROISSANCE_SORTS = 1.2
NB_POTIONS_CRAFTABLE_MAX = 6
COEF_COUT_VS_REWARD = 2
STOP_VALUE_TRESHOLD = 15
NB_SPELLS_CATEGORY = 5
NB_SPELLS_BY_CATEGORY = 2
# NB_LEARN_MAX = 4 + 2 + NB_SPELLS_BY_CATEGORY * NB_SPELLS_CATEGORY
NB_LEARN_MAX = 14
MIN_VALUE_GAIN_PYRAMIDE = 0.5
PYRAMIDE_NEGATIF_COEF = 1
PYRAMIDE_NEGATIF_COEF_BY_NEW_SORTS = 0.5
PYRAMIDE_SHOULD_STOP_AT_TOTAL_VALUE = 25


# REFAIRE L'HEURISTIQUE !!! Elle doit être une approximation du TEMPS restant. Et pas de la distance en gemmes !!!
# Faire l'achat des sorts en "pyramide" en sommant les coûts et les rewards. ==> le Temps devient la somme de ces
# scores divisé par le nombre de sorts ! <3
# Faire un chemin vers 2 potions d'un coup.
# Quand j'ai 5 potions, voir si il faut rush ou prendre la grande potion !


def debug(message: str, end="\n"):
    print(message, file=sys.stderr, flush=True, end=end)
    # pass


def read_input():
    return input()


class ActionType(Enum):
    SORT = "CAST"
    SORT_OPPONENT = "OPPONENT_CAST"
    POTION = "BREW"
    LEARN = "LEARN"
    REST = "REST"
    WAIT = "WAIT"


ALL_SPELLS = [
    np.array([-2, 2, 0, 0]),
    np.array([0, -2, 2, 0]),
    np.array([0, 0, -2, 2]),
    np.array([-3, 3, 0, 0]),
    np.array([0, -3, 3, 0]),
    np.array([0, 0, -3, 3]),
    np.array([-2, 0, 1, 0]),
    np.array([4, 0, 0, 0]),
    np.array([3, 0, 0, 0]),
    np.array([2, 1, 0, 0]),
    np.array([0, 2, 0, 0]),
    np.array([1, 0, 1, 0]),
    np.array([1, 1, 0, 0]),
    np.array([3, -1, 0, 0]),
    np.array([4, 1, -1, 0]),
    np.array([3, 0, 1, -1]),
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 1, 0]),
    np.array([-3, 0, 0, 1]),
    np.array([2, -2, 0, 1]),
    np.array([-4, 0, 2, 0]),
    np.array([2, 3, -2, 0]),
    np.array([2, 1, -2, 1]),
    np.array([3, -2, 1, 0]),
    np.array([2, -3, 2, 0]),
    np.array([2, 2, 0, -1]),
    np.array([-1, 0, -1, 1]),
    np.array([0, 2, -1, 0]),
    np.array([-3, 1, 1, 0]),
    np.array([0, 2, -2, 1]),
    np.array([1, -3, 1, 1]),
    np.array([0, 3, 0, -1]),
    np.array([0, -3, 0, 2]),
    np.array([1, 1, 1, -1]),
    np.array([1, 2, -1, 0]),
    np.array([-5, 0, 0, 2]),
    np.array([-4, 0, 1, 1]),
    np.array([0, 3, 2, -2]),
    np.array([1, 1, 3, -2]),
    np.array([-5, 0, 3, 0]),
    np.array([-2, 0, -1, 2]),
    np.array([0, 0, 2, -1])
]

SPELLS_0 = [
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 1, 0]),
]

SPELLS_1 = [
    np.array([4, 0, 0, 0]),
    np.array([3, 0, 0, 0]),
    np.array([2, 1, 0, 0]),
    np.array([1, 0, 1, 0]),
    np.array([1, 1, 0, 0]),
    np.array([0, 2, 0, 0]),
]

SPELLS_1_BIS = [
    np.array([3, -1, 0, 0]),
    np.array([4, 1, -1, 0]),
    np.array([3, 0, 1, -1]),
    np.array([3, -2, 1, 0]),
    np.array([2, 1, -2, 1]),
    np.array([2, -3, 2, 0]),
]
SPELLS_2 = [
    np.array([-2, 2, 0, 0]),
    np.array([-3, 3, 0, 0]),
    np.array([2, 3, -2, 0]),
    np.array([2, 2, 0, -1]),
    np.array([0, 2, -1, 0]),
    np.array([-3, 1, 1, 0]),
    np.array([0, 3, 0, -1]),
    np.array([1, 1, 1, -1]),
    np.array([1, 2, -1, 0]),
]

SPELLS_3 = [
    np.array([0, -2, 2, 0]),
    np.array([0, -3, 3, 0]),
    np.array([-2, 0, 1, 0]),
    np.array([-4, 0, 2, 0]),
    np.array([-3, 1, 1, 0]),
    np.array([0, 3, 2, -2]),
    np.array([-5, 0, 3, 0]),
]

SPELLS_4 = [
    np.array([0, 0, -2, 2]),
    np.array([0, 0, -3, 3]),
    np.array([-3, 0, 0, 1]),
    np.array([2, -2, 0, 1]),
    np.array([-1, 0, -1, 1]),
    np.array([0, 2, -2, 1]),
    np.array([0, -3, 0, 2]),
    np.array([-5, 0, 0, 2]),
    np.array([-4, 0, 1, 1]),
    np.array([-2, 0, -1, 2]),
]

ALL_SPELL_LISTS = [
    SPELLS_1,
    SPELLS_1_BIS,
    SPELLS_2,
    SPELLS_3,
    SPELLS_4,
]


class Retour:
    def __init__(self, action: Any, intent: str):
        self.action = action
        self.intent = intent

    def apply(self):
        if self.action == ActionType.REST.value:
            print(f"REST {self.intent}")
            debug(f"ACT = REST {self.intent}")
        elif self.action.type == ActionType.WAIT.value:
            print(f"WAIT {self.intent}")
            debug(f"ACT = WAIT {self.intent}")
        elif self.action.type == ActionType.SORT.value:
            print(f"CAST {self.action.id} {self.action.multiplicity} MUL={self.action.multiplicity} {self.intent}")
            debug(
                f"ACT = CAST {self.action.id} {self.action.multiplicity} MUL={self.action.multiplicity} {self.intent}")
        elif self.action.type == ActionType.POTION.value:
            print(f"BREW {self.action.id} {self.intent}")
            debug(f"ACT = BREW {self.action.id} {self.intent}")
        elif self.action.type == ActionType.LEARN.value:
            print(f"LEARN {self.action.id} {self.intent}")
            debug(f"ACT = LEARN {self.action.id} {self.intent}")
        else:
            debug(f"Don't know this action : {self.action} with intent {self.intent} !")

    @classmethod
    def construct(cls, something: Any, intent=None):
        if isinstance(something, str):
            intent = intent or "REST"
            return Retour(ActionType.REST.value, intent)
        elif isinstance(something, Learn):
            intent = intent or f"LEARN {something}"
            return Retour(something, intent)
        elif isinstance(something, Sort):
            intent = intent or f"{something}"
            return Retour(something, intent)
        elif isinstance(something, Potion):
            intent = intent or f"BREW {something}"
            return Retour(something, intent)
        else:
            debug(f"construct bizarre dans Retour : {something}")
            return None


def get_value_of_array(array: np.array) -> float:
    return np.sum(array * np.array([1, 2, 3, 4]))


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

    def get_nb_empty_spaces(self) -> int:
        return MAX_NB_ITEMS - np.sum(self.inv)

    def is_full(self) -> bool:
        return self.get_nb_items() >= MAX_NB_ITEMS

    def copy(self):
        return Inventory(self.inv)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.inv}"


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

        self.multiplicity = 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Sort({str(self.reward - self.cout)}({'O' if self.castable else 'X'}))"

    def is_castable(self, inventory: Inventory) -> bool:
        return self.castable \
               and all(inventory.inv - self.cout >= 0) \
               and np.sum(inventory.inv - self.cout + self.reward) <= MAX_NB_ITEMS

    def how_many_times_castable(self, inventory: Inventory) -> int:
        if not self.repeatable:
            return 1 if self.is_castable(inventory) else 0
        inv_copy = inventory.copy()
        for i in range(0, 10):
            if self.is_castable(inv_copy):
                inv_copy.update_inventory_with_sort(self)
            else:
                return i

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

    def set_multiplicity(self, nb_times: int):
        self.multiplicity = nb_times
        self.cout = self.cout * nb_times
        self.reward = self.reward * nb_times

    def is_in_list(self, list: List['np.array']) -> bool:
        for l in list:
            if all(self.reward - self.cout - l == 0):
                return True
        return False


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

    def rentability_on_buy(self, inventory: Inventory) -> int:
        if not self.is_buyable(inventory):
            return 0
        benef = self.tax_count - self.tome_index
        benef = min(benef, inventory.get_nb_empty_spaces())
        return benef

    def is_rentable_on_buy(self, inventory: Inventory) -> bool:
        if not self.is_buyable(inventory):
            return False
        return self.rentability_on_buy(inventory) > 0


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


def time_to_generate_gemmes(m: 'Model') -> 'np.array':
    pyramide = get_pyramide(m)
    times = len(m.sorts) / np.array([max(p, 1) for p in pyramide])
    times = [min(t, s) for t, s in zip(times, np.array([1, 2, 3, 4]))]
    return times


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
        return avancement_from_a_to_b(inventory.inv, self.cout)

    def get_first_retour(self) -> 'Retour':
        if self.path == []:
            return Retour(self, f"BREW POTION {self}")
        if self.path == None:
            return None
        return Retour.construct(self.path[-1])

    def get_distance(self) -> int:
        if self.path != None:
            return len(self.path) + 1
        else:
            return -1

    def is_brewable(self, inventory: Inventory):
        return all(inventory.inv - self.cout >= 0)


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


def avancement_from_a_to_b(a: np.array, b: np.array) -> float:
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


# def distance_from_a_to_b(a: np.array, b: np.array) -> float:
#     a = a.copy()
#     b = b.copy()
#     total = 0
#     diff = b - a
#     diff[diff < 0] = 0
#     return total


def compute_heuristique(n: 'Node', goal: 'Node', m: 'Model') -> float:
    # total = avancement_from_a_to_b(n.inventory.inv, goal.inventory.inv)
    diff = goal.inventory.inv - n.inventory.inv
    diff[diff < 0] = 0
    times_for_gemmes = time_to_generate_gemmes(m)
    total = np.sum(diff * times_for_gemmes)

    # from_sorts = compute_heuristique_sorts(n.sorts, m)
    # debug(f"heuristique from gemmes = {total} from sorts = {from_sorts}({len(n.sorts)})")
    # total += from_sorts
    sorts_castables = [s for s in n.sorts if isinstance(s, str) or s.castable]
    total -= (0.01 * len(sorts_castables))
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
        t = time.time()
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
        debug(f"t = {(t - time.time()) * 1000}ms")

    def get_voisins(self, goal: 'Node', m: 'Model', depth_max: int) -> List['Node']:
        if depth_max != -1 and self.cout >= depth_max:
            return []
        nodes = []
        for sort in self.get_possible_sorts():
            new_inventory = self.inventory.copy()
            new_inventory.update_inventory_with_sort(sort)
            new_sorts = update_sorts_with_sort(self.sorts, sort)
            node = Node(new_inventory, new_sorts, self.learns, self, goal, sort, m)
            nodes.append(node)

        usefull_learn = self.get_possible_learns()
        usefull_learn = [l for l in usefull_learn if l.rentability_on_buy(self.inventory) >= 1]
        for learn in usefull_learn:
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
            else:
                nb_casts_max = s.how_many_times_castable(self.inventory)
                for i in reversed(range(1, nb_casts_max + 1)):
                    sort_with_multiplicity = s.copy()
                    sort_with_multiplicity.set_multiplicity(i)
                    sort_castables.append(sort_with_multiplicity)
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


def a_star(start: Node, end: Node, m: 'Model', depth_max=-1) -> List[Sort]:
    opened = [start]
    closed = []

    while len(opened) > 0:
        if time.time() - m.debut_time >= SEUIL_TIME:
            debug(f"Out of time !")
            break

        current = opened[0]
        opened = opened[1:]

        if current.is_better_than(end):
            path = compute_path_backward(current)
            return path

        voisins = current.get_voisins(end, m, depth_max)
        # debug(f"time = {(time.time() - m.debut_time) * 1000}ms")

        for voisin in voisins:
            if voisin not in closed:
                opened = insert_in_opened(opened, voisin)

        closed.append(current)

    if depth_max != -1 and closed != []:
        best_depth = max([d.value for d in closed])
        best_depth_max = [d for d in closed if d.value == best_depth]
        return compute_path_backward(best_depth_max[0])

    if not time.time() - m.debut_time >= SEUIL_TIME:
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
    if m.nb_potions_to_craft > 1:
        potion_scores = [p.get_score(m) for p in m.potions]
        best_potion = m.potions[np.argmax([potion_scores])]
        return best_potion
    else:
        reachable = [p for p in m.potions if p.get_distance() != -1]
        if reachable != []:
            potion_distance = min([p.get_distance() for p in reachable])
            best_distance = [p for p in m.potions if p.get_distance() == potion_distance][0]
            return best_distance
        else:
            return None


def get_nb_needed_spells(m: 'Model') -> int:
    nb = NB_SPELLS_CATEGORY * NB_SPELLS_BY_CATEGORY
    for list in ALL_SPELL_LISTS:
        is_in_list = np.array([s.is_in_list(list) for s in m.sorts])
        nb_sorts_in = np.sum(is_in_list[is_in_list == True])
        nb -= min(nb_sorts_in, NB_SPELLS_BY_CATEGORY)
    return nb


def needed_spells(m: 'Model') -> List['np.array']:
    spell_lists = SPELLS_0
    for list in ALL_SPELL_LISTS:
        is_in_list = np.array([s.is_in_list(list) for s in m.sorts])
        nb_sorts_in = np.sum(is_in_list[is_in_list == True])
        if nb_sorts_in < NB_SPELLS_BY_CATEGORY:
            spell_lists.extend(list)
    return spell_lists


def get_first_learn(m: 'Model') -> 'Learn':
    return m.learns[0]


def get_pyramide_negatif_coef(m: 'Model') -> float:
    return 1 + PYRAMIDE_NEGATIF_COEF * PYRAMIDE_NEGATIF_COEF_BY_NEW_SORTS * (len(m.sorts) - NB_SORTS_INITIAUX)


def get_poids_gemmes(m: 'Model') -> 'np.array':
    nb_gemmes_total = np.sum([np.sum(p.cout) for p in m.potions])
    nb_gemmes = np.sum([p.cout for p in m.potions], axis=0)
    poids = nb_gemmes / nb_gemmes_total * NB_GEMMES
    return poids


def get_pyramide_value(pyramide: 'np.array', m: 'Model') -> float:
    poids_gemmes = get_poids_gemmes(m)
    negatif_coef = get_pyramide_negatif_coef(m)
    poids_gemmes[pyramide < 0] *= negatif_coef
    nb_gemmes = np.sum([p.cout for p in m.potions], axis=0)
    pyramide_cape = np.array([min(p, g) for p, g in zip(pyramide, nb_gemmes)])
    pyramide_value = np.sum(pyramide_cape * poids_gemmes, axis=0)
    return pyramide_value


def find_best_learn_for_pyramide(m: 'Model') -> 'Learn':
    pyramide = get_pyramide(m)
    new_pyramides_values = []
    for learn in m.learns:
        new_pyramide = pyramide + learn.reward - learn.cout
        new_pyramide_value = get_pyramide_value(new_pyramide, m)
        new_pyramides_values.append(new_pyramide_value)
    if len(new_pyramides_values) == 0:
        return None
    argmax_learn = np.argmax(new_pyramides_values)
    best_learn = m.learns[argmax_learn]
    return best_learn


def find_best_learn(m: 'Model') -> 'Learn':
    spells_to_look_at = needed_spells(m)
    debug(f"needed_spells = {spells_to_look_at}")

    for learn in m.learns:
        for s in spells_to_look_at:
            if all(learn.reward - learn.cout - s == 0):
                return learn
    return None


def apply_simple_algorithm(m: 'Model'):
    for sort in m.sorts:
        if sort.is_castable(m.me.inventory):
            return Retour(sort, f"DEFAULT SORT {sort}")
    return Retour(ActionType.REST.value, "DEFAULT REST :/")


def act(retour: Retour, m: 'Model') -> 'Model':
    if not retour:
        retour = apply_simple_algorithm(m)
    retour.apply()
    if isinstance(retour.action, Potion):
        m.nb_potions_to_craft -= 1
    return m


def go_for_value(m: 'Model') -> 'Retour':
    debug(f"GO FOR VALUE")
    if get_value_of_array(m.me.inventory.inv) >= STOP_VALUE_TRESHOLD or m.nb_potions_to_craft <= 1:
        return None
    end = Node(Inventory(np.array([10, 10, 10, 10])), [], learns=None, precedent=None, goal=None, sort_used=None, m=m)
    start = Node(m.me.inventory, m.sorts + [ActionType.REST.value], learns=m.learns, precedent=None, goal=end,
                 sort_used=None, m=m)
    path = a_star(start, end, m, depth_max=5)
    if path == None or path == []:
        return None
    else:
        return Retour(path[-1], f"VALUE {path[-1]}")


def go_for_potion(m: Model) -> 'Retour':
    debug(f"GO FOR POTION")
    potion_objectif = find_greedy_objectif(m)
    print(f"potion_objectif = (id={potion_objectif.id}, len={potion_objectif.get_distance()}, "
          f"score={potion_objectif.get_score(m)})", file=sys.stderr, flush=True)
    retour = potion_objectif.get_first_retour()
    if retour != None:
        if isinstance(retour.action, Learn):
            retour.intent = f"LEARN {potion_objectif.path[-1]} for {potion_objectif.path[-1].rentability_on_buy(m.me.inventory)}"
        elif isinstance(retour.action, Sort):
            retour.intent = f"{potion_objectif.path[-1]} for {potion_objectif}"
    return retour


def go_for_learn_spells_category(m: 'Model') -> 'Retour':
    debug(f"GO FOR LEARN SPELLS CATEGORY")
    nb_needed = get_nb_needed_spells(m)
    debug(f"Nb needed spells = {nb_needed}")
    if nb_needed <= NB_SPELLS_BY_CATEGORY or len(m.sorts) >= NB_LEARN_MAX:
        return None
    # if len(m.sorts) >= NB_LEARN_MAX:
    #     return None
    to_learn = find_best_learn(m)
    # to_learn = get_first_learn(m)
    if to_learn:
        return try_learn_learn(m, to_learn)
    debug(f"Need to learn, but don't know what to learn ! :'(")
    return None


def try_learn_learn(m: 'Model', to_learn: 'Learn') -> 'Retour':
    if to_learn:
        debug(f"Want to learn this {to_learn}")
        end = Node(Inventory(to_learn.achat_cout), [], learns=None, precedent=None, goal=None, sort_used=None, m=m)
        start = Node(m.me.inventory, m.sorts + [ActionType.REST.value], learns=m.learns, precedent=None, goal=end,
                     sort_used=None, m=m)
        path = a_star(start, end, m)
        if path == None:
            debug(f"Did not find a path to this {to_learn}")
            return None
        if path == []:
            m.sorts.append(to_learn.get_sort())
            debug(f"NOW WE HAVE {get_nb_needed_spells(m)} needed spells !")
            return Retour(to_learn, f"LEARN SORT {to_learn} !")
        debug(f"path to learn = {path}")
        return Retour(path[-1], f"LEARNING ... {to_learn}")
    debug(f"Need to learn, but don't know what to learn ! :'(")
    return None


def should_stop_learn_pyramide(to_learn: 'Learn', m: 'Model') -> bool:
    old_pyramide = get_pyramide(m)
    old_value = get_pyramide_value(old_pyramide, m)
    new_pyramide = old_pyramide - to_learn.cout + to_learn.reward
    new_value = get_pyramide_value(new_pyramide, m)
    gain_value = new_value - old_value
    debug(f"Value expected learn spell = {gain_value}")
    not_enought_gain = gain_value < MIN_VALUE_GAIN_PYRAMIDE
    # old_pyramide_total_value_high_enough = np.sum(old_pyramide, axis=0) >= PYRAMIDE_SHOULD_STOP_AT_TOTAL_VALUE
    everything_positif = all(old_pyramide >= 0)
    # return not everything_positif and (not_enought_gain or old_pyramide_total_value_high_enough)
    has_enought_sorts = len(m.sorts) > NB_LEARN_MAX
    if not everything_positif:
        return False
    return not_enought_gain or has_enought_sorts


def get_pyramide(m: 'Model') -> 'np.array':
    pyramide = np.zeros(NB_GEMMES)
    sorts = m.sorts.copy()
    sorted(sorts, key=lambda s: 1 if s.repeatable else 0)
    while len(sorts) > 0:
        has_changed = False
        for sort in sorts:
            while all(pyramide - sort.cout >= 0):
                pyramide += sort.reward - sort.cout
                has_changed = True
                if sort in sorts:
                    sorts.remove(sort)
                if not sort.repeatable:
                    break
        if not has_changed:
            break
    return pyramide


def go_for_learn_pyramide(m: 'Model') -> 'Retour':
    debug(f"GO FOR LEARN PYRAMIDE")
    debug(f'Pyramide = {get_pyramide(m)} (negative coef = {get_pyramide_negatif_coef(m)})')
    to_learn = find_best_learn_for_pyramide(m)
    if not to_learn or should_stop_learn_pyramide(to_learn, m):
        return None
    retour = try_learn_learn(m, to_learn)
    debug(f"3 time = {(time.time() - m.debut_time) * 1000}ms")
    return retour


def if_can_brew_brew(m: Model) -> 'Retour':
    debug(f"IF CAN BREW POTION")
    brewables = [p for p in m.potions if p.is_brewable(m.me.inventory)]
    if brewables == []:
        return None
    max_price = max([b.price for b in brewables])
    max_brewables = [b for b in brewables if b.price == max_price]
    best_brewable = max_brewables[0]
    return Retour(best_brewable, f"INSTA_BUY {best_brewable}")


def think(m: 'Model') -> 'Retour':
    retour = if_can_brew_brew(m)
    debug(f"IF CAN BREW POTION time = {(time.time() - m.debut_time) * 1000}ms")
    if retour:
        return retour
    # retour = go_for_learn_spells_category(m)
    retour = go_for_learn_pyramide(m)
    debug(f"GO FOR LEARN PYRAMIDE time = {(time.time() - m.debut_time) * 1000}ms")
    if retour:
        return retour
    retour = go_for_value(m)
    debug(f"GO FOR VALUE time = {(time.time() - m.debut_time) * 1000}ms")
    if retour:
        return retour
    retour = go_for_potion(m)
    debug(f"GO FOR POTION time = {(time.time() - m.debut_time) * 1000}ms")
    return retour


def run():
    current_round = 0
    nb_potions_to_craft = NB_POTIONS_CRAFTABLE_MAX
    while True:
        m = Model.read(nb_potions_to_craft)
        m.debut_time = time.time()

        retour = think(m)
        m = act(retour, m)

        current_round += 1
        nb_potions_to_craft = m.nb_potions_to_craft
        print(f"round = {current_round} (time={(time.time() - m.debut_time) * 1000}ms)", file=sys.stderr, flush=True)


if __name__ == '__main__':
    if os.path.exists('in.txt'):
        sys.stdin = open('in.txt', 'r', newline='\n')
    run()
