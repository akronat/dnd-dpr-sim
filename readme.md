# D&D Damage Per Round Simulator

Performs simulations of attacks based on custom definitions. Simulations are defined with the following form:
```
<name>@<level(s)>: <expression>
```
Where the expression is a custom definition such as in the following:
```
GreatAxe@1: 3+PB =atk> 1D12+3
GreatSword@1: 3+PB =atk> 2D6+3
Dual Wield@1: (3+PB =atk> 1D6+3) + (3+PB =atk> 1D6)
GreatSword with GreatWeaponMaster@1: 3+PB-5 =atk> 2D6+3+10
```
In the above examples, 3 is the attacking attribute modifier, while PB is the proficiency bonus (automatically calculated based on level).
The uppercase D in the 1D6 damage roll means that the number of dice is doubled on a crit.

Multiple attacks can be easily defined with # notation:
```
GreatAxe@5: 2#(4+PB =atk> 1D12+3)
GreatSword@5: 2#(4+PB =atk> 2D6+3)
Dual Wield@5: 2#(4+PB =atk> 1D6+3) + (3+PB =atk> 1D6)
```

More advanced definitions are also possible...
```
GreatAxe with Great Weapon Fighting@1: 3+PB =atk> CM#(1d12 @rrlte: 2) + 3
GreatSword with Great Weapon Fighting@1: 3+PB =atk> CM#(2#(1d6 @rrlte: 2)) + 3
GreatAxe with Advantage@1: 3+PB =atk:adv> 1D12+3
GreatAxe with Disadvantage@1: 3+PB =atk:dis> 1D12+3
GreatAxe that crits on a 19@1: 3+PB =atk:19> 1D12+3
GreatAxe with Advantage that crits on a 19@1: 3+PB =atk:19:adv> 1D12+3
Dual Wield with a Sneak Attack@1: ($a := (3+PB =atk> 1D6+3 + 1D6)) + (3+PB =atk> 1D6 + ($a<=0 => 1D6)) + $a
Sorlock with Spirit Guardians and Booming Blade@7: (8+4+PB =sav:50> 3d8) + (4+PB =atk> 2D8+4)
```
