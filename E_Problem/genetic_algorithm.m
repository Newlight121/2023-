function [BestSol, BestCosts] = genetic_algorithm(CostFunction, nVar, VarSize, VarMin, VarMax, MaxIt, nPop, pc, nc, pm,nm, mu, options)
% 遗传算法
% 参数说明：
% CostFunction - cost function
% nVar - number of decision variables
% VarSize - decision variables size
% VarMin - decision variables lower bound
% VarMax - decision variables upper bound
% MaxIt - maximum number of iterations
% nPop - population size
% pc - crossover rate
% pm - mutation rate
% mu - mutation factor
% Output:
% BestSol - best solution
% BestCosts - best cost of each iteration

% Initialize empty individual struct
%% 初始化种群

empty_individual.Position = [];
empty_individual.Cost = [];

pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop(i).Cost = CostFunction(pop(i).Position);
end

% 最优解的记录
BestSol = pop(1);

% 历史最优解的记录
BestCosts = zeros(MaxIt, 1);

%% 输出种群初始化信息
if (options.Display=="iter")
disp(' ');
disp('遗传算法正在初始化种群 ...');
disp(['初始种群大小： ' num2str(nPop)]);
disp(['决策变量数量： ' num2str(nVar)]);
disp(['最大迭代次数： ' num2str(MaxIt)]);
disp(['交叉概率： ' num2str(pc)]);
disp(['变异概率： ' num2str(pm)]);
%% 遗传算法主体

disp(' ');
disp('遗传算法正在求解中 ...');
end

for it = 1:MaxIt

    % 交叉
    popc = repmat(empty_individual, nc/2, 2);
    for k = 1:nc/2
        i1 = randi([1 nPop]);
        i2 = randi([1 nPop]);
        p1 = pop(i1);
        p2 = pop(i2);
        [popc(k,1).Position, popc(k,2).Position] = ...
            SinglePointCrossover(p1.Position, p2.Position);
        popc(k,1).Cost = CostFunction(popc(k,1).Position);
        popc(k,2).Cost = CostFunction(popc(k,2).Position);
    end
    popc = popc(:);

    % 变异
    popm = repmat(empty_individual, nm, 1);
    for k = 1:nm
        i = randi([1 nPop]);
        popm(k).Position = Mutate(pop(i).Position, mu, VarMin, VarMax);
        popm(k).Cost = CostFunction(popm(k).Position);
    end

    % 合并种群
%     pop = [popc;popm];
    % 选择
    pop_all = [pop; popc; popm];
    costs_all = [pop_all.Cost];
    [costs_all, inds] = sort(costs_all);
    pop_all = pop_all(inds);
    pop = pop_all(1:nPop);

    % 记录历史最优解
    BestSol = pop(1);
    BestCosts(it) = BestSol.Cost;

    % 输出当前迭代的结果
    if (options.Display=="iter")
        disp(['迭代次数： ' num2str(it) ',  最优解： ' num2str(BestCosts(it))]);
    end
end

end

% Single-point crossover operator
function [c1, c2] = SinglePointCrossover(p1, p2)

nVar = numel(p1);
c = randi([1 nVar-1]);
c1 = [p1(1:c) p2(c+1:end)];
c2 = [p2(1:c) p1(c+1:end)];
end

function y = Mutate(x, mu, VarMin, VarMax)

nVar = numel(x);
nmu = ceil(mu*nVar);
j = randi([1 nVar], [nmu 1]);
sigma = 0.1*(VarMax-VarMin);
y = x;
y(j) = x(j) + sigma*randn(size(j));
y = max(y, VarMin);
y = min(y, VarMax);
end
