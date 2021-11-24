#include "Unit.h"

void Unit::initRandom(int stateSize, int stateHidden, int connectionValues, int connectionHidden, std::mt19937 &rng, float initMinWeight, float initMaxWeight) {
    this->stateSize = stateSize;
    this->stateHidden = stateHidden;
    this->connectionValues = connectionValues;
    this->connectionHidden = connectionHidden;

    stateUpdater.initRandom({ stateSize + connectionValues * 3 + 3, stateHidden, stateSize }, rng, initMinWeight, initMaxWeight);

    ff.initRandom({ stateSize + connectionValues, connectionHidden, connectionValues }, rng, initMinWeight, initMaxWeight);
    lat.initRandom({ stateSize + connectionValues, connectionHidden, connectionValues }, rng, initMinWeight, initMaxWeight);
    fb.initRandom({ stateSize + connectionValues, connectionHidden, connectionValues }, rng, initMinWeight, initMaxWeight);
}

void Unit::initFromParents(const Unit &parent1, const Unit &parent2, std::mt19937 &rng) {
    this->stateSize = parent1.stateSize;
    this->stateHidden = parent1.stateHidden;
    this->connectionValues = parent1.connectionValues;
    this->connectionHidden = parent1.connectionHidden;

    stateUpdater.initFromParents(parent1.stateUpdater, parent2.stateUpdater, rng);

    ff.initFromParents(parent1.ff, parent2.ff, rng);
    lat.initFromParents(parent1.lat, parent2.lat, rng);
    fb.initFromParents(parent1.fb, parent2.fb, rng);
}

void Unit::mutate(float chance, float pertWeight, std::mt19937 &rng) {
    stateUpdater.mutate(chance, pertWeight, rng);

    ff.mutate(chance, pertWeight, rng);
    lat.mutate(chance, pertWeight, rng);
    fb.mutate(chance, pertWeight, rng);
}
