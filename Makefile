.PHONY: help setup figures bandit qlearn dqn ppo offline notebooks

help:
	@echo "Targets:"
	@echo "  make setup     # create .venv + install deps"
	@echo "  make figures   # generate figures/*.png (requires deps)"
	@echo "  make bandit    # run bandit demo"
	@echo "  make qlearn    # run tabular q-learning demo"
	@echo "  make dqn       # run dqn demo"
	@echo "  make ppo       # run ppo demo"
	@echo "  make offline   # generate offline dataset + run OPE toy"

setup:
	bash scripts/setup.sh

figures:
	. .venv/bin/activate && python3 code/utils/make_figures.py

bandit:
	. .venv/bin/activate && python3 code/bandit/run_bandit.py

qlearn:
	. .venv/bin/activate && python3 code/gridworld/train_q_learning.py

dqn:
	. .venv/bin/activate && python3 code/dqn/train_dqn_grid.py

ppo:
	. .venv/bin/activate && python3 code/ppo/train_ppo_grid.py

offline:
	. .venv/bin/activate && python3 code/offline/make_offline_dataset.py && python3 code/offline/ope_is_dr.py
