import sys
import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from typing import List, Any, Tuple


# action_id: the unique ID of this spell or recipe
# action_type: in the first league: BREW; later: CAST, OPPONENT_CAST, LEARN, BREW
# delta_0: tier-0 ingredient change
# delta_1: tier-1 ingredient change
# delta_2: tier-2 ingredient change
# delta_3: tier-3 ingredient change
# price: the price in rupees if this is a potion
# tome_index: in the first two leagues: always 0; later: the index in the tome if this is a tome spell,
# equal to the read-ahead tax
# tax_count: in the first two leagues: always 0; later: the amount of taxed tier-0 ingredients you gain from
# learning this spell
# castable: in the first league: always 0; later: 1 if this is a castable player spell
# repeatable: for the first two leagues: always 0; later: 1 if this is a repeatable player spell


class ActionType(Enum):
    SORT = "CAST"
    SORT_OPPONENT = "OPPONENT_CAST"
    POTION = "BREW"


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
        self.reward = self.cout
        self.reward[self.reward < 0] = 0
        self.cout[self.cout > 0] = 0
        self.cout = - self.cout

    def is_castable(self, inventory: np.array) -> bool:
        return all(inventory - self.cout >= 0)


class Gemme:
    @classmethod
    def how_to_get_to_gemmes(cls, inventory: np.array, sorts: List[Sort], goal: np.array) -> List[Sort]:  # sorts to use
        if all(inventory - goal >= 0):
            return []

        list_sort_to_use = []
        for i, sort in enumerate(sorts):
            if sort.is_castable(inventory):
                sorts_after_use = sorts.copy()
                sorts_after_use[i] = replace(sort, castable=False)
                sorts_to_use = cls.how_to_get_to_gemmes(inventory + sorts[0].reward, sorts_after_use, goal)
                sorts_to_use.append(sort)
                list_sort_to_use.append(sorts_to_use)

        min_length = min([len(l) for l in list_sort_to_use])
        best_lists = [l for l in list_sort_to_use if len(l) == min_length]
        # best_inventory = prendre celle qui a le meilleur inventaire
        return best_lists[0]


class Action:
    @staticmethod
    def read():
        action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable, repeatable \
            = input().split()
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
        self.price = price
        self.castable = castable
        self.tome_index = tome_index
        self.tax_count = tax_count
        self.repeatable = repeatable

    @staticmethod
    def read():
        action_id, action_type, delta_0, delta_1, delta_2, delta_3, price, tome_index, tax_count, castable, repeatable \
            = input().split()
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

    def get_score(self, inventory: np.array, sorts: List[Sort], goal: np.array) -> float:
        return self.difficulty_to_get(inventory, sorts, goal) / self.price

    def difficulty_to_get(self, inventory: np.array, sorts: List[Sort], goal: np.array) -> int:
        return len(Gemme.how_to_get_to_gemmes(inventory=inventory, sorts=sorts, goal=goal))


class Player:
    def __init__(self, inv0: int, inv1: int, inv2: int, inv3: int, score: int):
        self.inv = np.array([inv0, inv1, inv2, inv3])
        self.score = score

    def can_buy(self, potion: Potion) -> bool:
        return np.all(self.inv - potion.cout)

    def buy(self, potion: Potion) -> 'Player':
        new_player = Player(self.inv[0], self.inv[1], self.inv[2], self.inv[3], self.score)
        new_player.inv -= potion.cout
        new_player.score += potion.price
        return new_player

    @staticmethod
    def read():
        return Player(*[int(i) for i in input().split()])


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
    print(f"best_indices = {best_potions}, best_player = {best_player}", file=sys.stderr, flush=True)
    return best_potions[0]


class Model:
    def __init__(self, potions: List[Potion], sorts: List[Sort], opp_sorts: List[Sort], me: Player, opp: Player):
        self.potions = potions
        self.sorts = sorts
        self.opp_sorts = opp_sorts
        self.me = me
        self.opp = opp

    @staticmethod
    def read():
        nb_objects = int(input())
        actions = [Action.read() for _ in range(nb_objects)]
        potions = [a for a in actions if a.type == ActionType.POTION]
        sorts = [a for a in actions if a.type == ActionType.SORT]
        opp_sorts = [a for a in actions if a.type == ActionType.SORT_OPPONENT]
        me = Player.read()
        opp = Player.read()
        return Model(potions, sorts, opp_sorts, me, opp)


def find_greedy_objectif(m: Model) -> Potion:
    potions_scores = [p.get_score() for p in m.potions]
    pass


def run():
    current_round = 0
    nb_potions_to_craft = 3
    while True:
        m = Model.read()
        
        potion_objectif = find_greedy_objectif(m)

        best_potion = compute_best(m.potions, m.me, m.opp, nb_potions_to_craft)

        print(f"BREW {best_potion.id}")

        current_round += 1


if __name__ == '__main__':
    run()
