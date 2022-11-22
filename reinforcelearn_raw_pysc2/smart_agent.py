from pysc2.lib import units
from pysc2.lib.features import ScoreCumulative

from reinforcelearn_raw_pysc2.base_agent import Agent
from reinforcelearn_raw_pysc2.q_learn_table import QLearningTable


class SmartAgent(Agent):
    KILL_UNIT_REWARD = 0.2
    KILL_BUILDING_REWARD = 0.5

    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.actions)
        self.base_top_left = None
        self.previous_action = None
        self.previous_state = None
        self.name = 'SmartAgent'
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_action = None
        self.previous_state = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))
        action = self.qtable.choose_action(state)
        killed_unit_score = obs.observation['score_cumulative'][ScoreCumulative.killed_value_units]
        killed_building_score = obs.observation['score_cumulative'][ScoreCumulative.killed_value_structures]

        if self.previous_action is not None:
            reward = 0
            if killed_unit_score > self.previous_killed_unit_score:
                reward += self.KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += self.KILL_BUILDING_REWARD

            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              reward,
                              'terminal' if obs.last() else state)
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = state
        self.previous_action = action
        return getattr(self, action)(obs)