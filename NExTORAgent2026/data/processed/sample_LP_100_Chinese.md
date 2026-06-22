# prob_001

## question

假设某种动物每天至少需要 $700 \mathrm{~g}$ 蛋白质、$30 \mathrm{~g}$ 矿物质和 $100 \mathrm{mg}$ 维生素。现有 5 种饲料可供选择，各种饲料的营养含量及每千克价格如表 1-5 所示：  
试建立线性规划模型，在满足该动物生长需求的前提下，使所选饲料的成本最低。

表 1-6  

| 饲料 | 蛋白质 (g) | 矿物质 (g) | 维生素 (mg) | 价格 (¥/kg) | 饲料 | 蛋白质 (g) | 矿物质 (g) | 维生素 (mg) | 价格 (¥/kg) |
|------|-------------|-------------|-------------|-------------|------|-------------|-------------|-------------|-------------|
| 1    | 3           | 1           | 0.5         | 0.2         | 4    | 6           | 2           | 2           | 0.3         |
| 2    | 2           | 0.5         | 1           | 0.7         | 5    | 18          | 0.5         | 0.8         | 0.8         |
| 3    | 1           | 0.2         | 0.2         | 0.4         |      |             |             |             |             |

### Other Details
- **description**: Suppose a certain animal needs at least $700 \mathrm{~g}$ of protein, $30 \mathrm{~g}$ of minerals, and $100 \mathrm{mg}$ of vitamins daily. There are 5 types of feed available, and the nutritional content and price per gram of each type of feed are shown in Table 1-5:
Try to formulate a linear programming model that meets the animal's growth needs while minimizing the cost of selecting the feed.
Table 1-6
| Feed | Protein (g) | Minerals (g) | Vitamins (mg) | Price (¥/kg) | Feed | Protein (g) | Minerals (g) | Vitamins (mg) | Price (¥/kg) |
|------|-------------|--------------|---------------|--------------|------|-------------|--------------|---------------|--------------|
| 1    | 3           | 1            | 0.5           | 0.2          | 4    | 6           | 2            | 2             | 0.3          |
| 2    | 2           | 0.5          | 1             | 0.7          | 5    | 18          | 0.5          | 0.8           | 0.8          |
| 3    | 1           | 0.2          | 0.2           | 0.4          |      |             |              |               |              |
- **ground_truth**: 32.43589743589744
- **problem_type**: LP
- **problem_size**: Toy
- **index**: IndustryOR_prob_029
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_029\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_029\code.py

---

# prob_002

## question

汤姆和杰瑞刚刚在阳光谷买了一座农场，他们正在考虑用它来种植玉米、小麦、大豆和高粱。种植玉米的每英亩利润为 1500 美元，种植小麦的每英亩利润为 1200 美元，种植大豆的每英亩利润为 1800 美元，种植高粱的每英亩利润为 1600 美元。为了使他们的利润最大化，他们应当为每种作物分配多少英亩的土地？汤姆和杰瑞的农场总面积为 100 英亩。

用于种植玉米的土地面积必须至少是用于种植小麦的土地面积的两倍。

用于种植大豆的土地面积必须至少是用于种植高粱的土地面积的一半。

用于种植小麦的土地面积必须是用于种植高粱的土地面积的三倍。

### Other Details
- **description**: Tom and Jerry just bought a farm in Sunshine Valley, and they are considering using it to plant corn, wheat, soybeans, and sorghum. The profit per acre for planting corn is $1500, the profit per acre for planting wheat is $1200, the profit per acre for planting soybeans is $1800, and the profit per acre for planting sorghum is $1600. To maximize their profit, how many acres of land should they allocate to each crop? Tom and Jerry’s farm has a total area of 100 acres.

The land area used for planting corn must be at least twice the land area used for planting wheat.

The land area used for planting soybeans must be at least half the land area used for planting sorghum.

The land area used for planting wheat must be three times the land area used for planting sorghum.
- **ground_truth**: 180000
- **problem_type**: LP
- **problem_size**: Toy
- **index**: IndustryOR_prob_012
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_012\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_012\code.py

---

# prob_003

## question

有两种产品 $ \mathrm{A} $ 和 $ \mathrm{B} $，它们都需要经过两个连续的化学反应工序。每件产品 $ \mathrm{A} $ 在第一道工序上需要 2 小时，在第二道工序上需要 3 小时；每件产品 $ \mathrm{B} $ 在第一道工序上需要 3 小时，在第二道工序上需要 4 小时。第一道工序可用的时间为 16 小时，第二道工序可用的时间为 24 小时。

每生产 1 件产品 $ \mathrm{B} $，会同时副产 2 件副产品 $ \mathrm{C} $，且无需额外成本。副产品 $ \mathrm{C} $ 最多可销售 5 件，超过部分必须以每件 2 元的成本进行处理。

每销售 1 件产品 $ \mathrm{A} $ 的利润为 4 元，每销售 1 件产品 $ \mathrm{B} $ 的利润为 10 元，每销售 1 件副产品 $ \mathrm{C} $ 的利润为 3 元。

为使总利润最大，建立该问题的线性规划模型。

### Other Details
- **description**: There are $\mathrm{A}$ and $\mathrm{B}$ two products, both requiring two successive chemical reaction processes. Each unit of product $\mathrm{A}$ needs 2 hours for the first process and 3 hours for the second process. Each unit of product $\mathrm{B}$ needs 3 hours for the first process and 4 hours for the second process. Available time for the first process is 16 hours, and available time for the second process is 24 hours.

For each unit of product $\mathrm{B}$ produced, 2 units of by-product $\mathrm{C}$ are generated simultaneously, requiring no additional cost. By-product $\mathrm{C}$ can be sold up to 5 units, and the rest must be disposed of at a cost of 2 yuan per unit.

Each unit of product $\mathrm{A}$ sold yields a profit of 4 yuan, each unit of product $\mathrm{B}$ yields a profit of 10 yuan, and each unit of by-product $\mathrm{C}$ sold yields a profit of 3 yuan.

In order to maximize total profit, establish the linear programming model for this problem.
- **ground_truth**: 57.0
- **problem_type**: LP
- **problem_size**: Toy
- **index**: IndustryOR_prob_074
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_074\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_074\code.py

---

# prob_004

## question

一位富有的贵族去世，留下了如下遗产：

- 一幅卡耶博特的画：$25000  
- 一尊戴克里先的半身像：$5000  
- 一只元代中国花瓶：$20000  
- 一辆保时捷 911：$40000  
- 三颗钻石：每颗 $12000  
- 一张路易十五时期的沙发：$3000  
- 两只极为名贵的杰克罗素赛犬：每只 $3000（遗嘱规定它们不能被分开）  
- 一件公元 200 年的雕塑：$10000  
- 一艘帆船：$15000  
- 一辆哈雷戴维森摩托车：$10000  
- 一件曾属于加富尔的家具：$13000  

这些遗产必须在两个儿子之间进行分配。如何使两部分的价值差最小？

### Other Details
- **description**: A wealthy noble passed away, leaving the following inheritance:

- A painting by Caillebotte: $25000
- A bust of Diocletian: $5000
- A Yuan dynasty Chinese vase: $20000
- A 911 Porsche: $40000
- Three diamonds: each $12000
- A Louis XV sofa: $3000
- Two very precious Jack Russell racing dogs: each $3000 (will stipulates they must not be separated)
- A sculpture from 200 AD: $10000
- A sailing boat: $15000
- A Harley Davidson motorcycle: $10000
- A piece of furniture once belonging to Cavour: $13000,

which must be shared between two sons. How to minimize the difference in value between the two parts?
- **ground_truth**: 1000
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_032
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_032\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_032\code.py

---

# prob_005

## question

工厂生产三种产品：Ⅰ、Ⅱ和Ⅲ。每种产品都需要经过两个加工阶段：A 和 B。假设工厂有两类设备可以完成 A 加工阶段，分别记为 A1 和 A2；有两类设备可以完成 B 加工阶段，分别记为 B1 和 B2。产品Ⅰ在 A 加工阶段可以在任意一种 A 类设备上加工，但在 B 加工阶段只能在 B1 设备上加工；产品Ⅲ只能在 A2 和 B2 设备上加工。各产品的单位加工时间、原材料成本、销售价格，各类设备的有效机时以及设备满负荷运转成本如表 1‑18 所示。要求安排最优生产计划，使工厂利润最大。

表 1‑18  
| 设备                      | 产品Ⅰ | 产品Ⅱ | 产品Ⅲ | 有效机时        | 设备满负荷运转成本（元） |
|---------------------------|--------|--------|--------|-----------------|-------------------------|
| A1                        | 5      | 10     |        | 10000           | 321                     |
| A2                        | 7      | 9      | 12     | 4000            | 250                     |
| B1                        | 6      | 8      |        | 7000            | 783                     |
| B2                        | 4      |        | 11     | 4000            | 200                     |
| B3                        | 7      |        |        |                 |                         |
| 原材料成本（元/件）       | 0.25   | 0.35   | 0.50   |                 |                         |
| 单价（元/件）             | 1.25   | 2.00   | 2.80   |                 |                         |

### Other Details
- **description**: A factory produces three types of products, I, II, and III. Each product requires two processing stages, A and B. It is assumed that the factory has two types of equipment to complete the A processing stage, denoted as A1 and A2; and two types of equipment to complete the B processing stage, denoted as B1 and B2. Product I can be processed on any type of A equipment, but for the B processing stage, it can only be processed on B1 equipment; Product III can only be processed on A2 and B2 equipment. The processing time per piece, raw material cost, sales price of products, effective machine hours for various equipment, and costs of equipment at full capacity are given in Table 1-18. The requirement is to arrange the optimal production plan so that the factory's profit is maximized.

Table 1-18
| Equipment   | Product I | Product II | Product III | Effective Machine Hours | Cost of Equipment at Full Capacity (Yuan) |
|-------------|-----------|------------|-------------|------------------------|-------------------------------------------|
| A1          | 5         | 10         |             | 10000                 | 321                                       |
| A2          | 7         | 9          | 12          | 4000                  | 250                                       |
| B1          | 6         | 8          |             | 7000                  | 783                                       |
| B2          | 4         |            | 11          | 4000                  | 200                                       |
| B3          | 7         |            |             |                       |                                           |
| Raw Material Cost (Yuan/Piece) | 0.25      | 0.35       | 0.50       |                        |                                           |
| Unit Price (Yuan/Piece)        | 1.25      | 2.00       | 2.80       |                        |                                           |
- **ground_truth**: 712.875
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_030
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_030\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_030\code.py

---

# prob_006

## question

一家公司使用钢材和铝材作为原材料生产两种产品（A 和 B）。生产 1 单位产品 A 需要 6 千克钢材、8 千克铝材、11 小时劳动，并带来 5000 元利润（不含工人加班费）。生产 1 单位产品 B 需要 12 千克钢材、20 千克铝材、24 小时劳动，并带来 11000 元利润（不含工人加班费）。公司目前拥有 200 千克钢材、300 千克铝材和 300 小时的劳动时间。如果工人需要加班，加班费为每小时 100 元。请制定一个生产计划，使公司的利润最大化并使工人加班时间最少。

### Other Details
- **description**: A company uses steel and aluminum as raw materials to produce two products (A and B). A single unit of product A requires 6 kg of steel, 8 kg of aluminum, 11 hours of labor, and yields a profit of 5000 yuan (excluding worker overtime pay). A single unit of product B requires 12 kg of steel, 20 kg of aluminum, 24 hours of labor, and yields a profit of 11000 yuan (excluding worker overtime pay). The company currently has 200 kg of steel, 300 kg of aluminum, and 300 hours of labor available. If workers need to work overtime, the overtime pay is 100 yuan per hour. Please develop a production plan to maximize the company's profit and minimize worker overtime.
- **ground_truth**: 165900.0
- **problem_type**: MILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_097
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_097\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_097\code.py

---

# prob_007

## question

一家公司计划生产 3 种产品 $A_{1}, A_{2}, A_{3}$。每月可生产 22 天。下表给出了最大需求（单位 $=100 \mathrm{~kg}$）、价格（$\$/100 \mathrm{Kg}$）、生产成本（每 100Kg 产品）以及生产配额（当所有生产线全部用于该产品时，每天最多可生产的 100kg 单位数量）。

| 产品 | $A_{1}$ | $A_{2}$ | $A_{3}$ |
| :---: | :---: | :---: | :---: |
| 最大需求 | 5300 | 4500 | 5400 |
| 销售价格 | $124$ | $109$ | $115$ |
| 生产成本 | $73.30$ | $52.90$ | $65.40$ |
| 生产配额 | 500 | 450 | 550 |

生产线的固定启用成本如下：

| 产品 | $A_{1}$ | $A_{2}$ | $A_{3}$ |
| :---: | :---: | :---: | :---: |
| 启用成本 | $170000$ | $150000$ | $100000$ |

最小生产批量：

$$
\begin{array}{c|ccc}
\text{产品} & A_{1} & A_{2} & A_{3} \\
\hline
\text{最小批量} & 20 & 20 & 16
\end{array}
$$

请建立一个运筹学模型，在考虑固定启用成本和最小生产批量约束的前提下，确定一个使总收益最大化的生产计划。

### Other Details
- **description**: A company plans to produce 3 types of products $A_{1}, A_{2}, A_{3}$. It can produce for 22 days in a month. The following table gives the maximum demand (unit $=100 \mathrm{~kg}$), price ($\$ / 100 \mathrm{Kg}$), production cost (per 100Kg product), and production quota (the maximum number of 100kg units that can be produced in one day if all production lines are devoted to this product).

| Product | $A_{1}$ | $A_{2}$ | $A_{3}$ |
| :---: | :---: | :---: | :---: |
| Maximum Demand | 5300 | 4500 | 5400 |
| Selling Price | $124$ | $109$ | $115$ |
| Production Cost | $73.30$ | $52.90$ | $65.40$ |
| Production Quota | 500 | 450 | 550 |

The fixed activation cost of the production line is as follows:

| Product | $A_{1}$ | $A_{2}$ | $A_{3}$ |
| :---: | :---: | :---: | :---: |
| Activation Cost | $170000$ | $150000$ | $100000$ |

Minimum production batch:

$$
\begin{array}{c|ccc}
Product & A_{1} & A_{2} & A_{3} \\
\hline
Minimum Batch & 20 & 20 & 16
\end{array}
$$

Please formulate an operations research model to determine a production plan that maximizes total revenue while accommodating fixed activation costs and minimum production batch constraints.
- **ground_truth**: 270290.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_021
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_021\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_021\code.py

---

# prob_008

## question

一家饮料厂生产某种饮料以满足市场需求。根据市场预测，工厂销售部门确定了未来 4 周该饮料的需求量。计划部门根据工厂的实际情况给出了未来 4 周的生产能力和生产成本，如表 1 所示。每周在满足需求后，如有饮料剩余，则需按每 1000 箱饮料每周 0.2 千元的标准支付库存成本。问应如何安排 4 周的生产计划，在满足各周市场需求的前提下，使 4 周的总成本（生产成本与库存成本之和）最小？

表 1 饮料生产与需求数据：

\begin{tabular}{c|c|c|c}
\hline 
周别 & 需求量/1000箱 & 生产能力/1000箱 & 单位生产成本/1000箱·千元 \\
\hline 
1 & 15 & 30 & 5.0 \\
\hline 
2 & 25 & 40 & 5.1 \\
\hline 
3 & 35 & 45 & 5.4 \\
\hline 
4 & 25 & 20 & 5.5 \\
\hline 
合计 & 100 & 135 & \\
\hline
\end{tabular}

### Other Details
- **description**: A beverage factory produces a kind of beverage to meet market demand. According to market forecasts, the sales department of the factory has determined the demand for the beverage for the next 4 weeks. The planning department, based on the actual situation of the factory, has provided the production capacity and production cost for the next 4 weeks, as shown in Table 1. When there is a surplus of beverages after meeting the demand each week, a storage cost of 0.2 thousand yuan per week per thousand boxes of beverages needs to be paid. How should the production plan be arranged to minimize the total cost (the sum of production cost and storage cost) over the four weeks while meeting the weekly market demand?

Table 1 Beverage Production and Demand Data:

\begin{tabular}{c|c|c|c}
\hline 
Week & Demand/1000 boxes & Production Capacity/1000 boxes & Cost per 1000 boxes/1000 yuan \\
\hline 
1 & 15 & 30 & 5.0 \\
\hline 
2 & 25 & 40 & 5.1 \\
\hline 
3 & 35 & 45 & 5.4 \\
\hline 
4 & 25 & 20 & 5.5 \\
\hline 
Total & 100 & 135 & \\
\hline
\end{tabular}
- **ground_truth**: 528.0
- **problem_type**: LP
- **problem_size**: Small
- **index**: IndustryOR_prob_094
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_094\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_094\code.py

---

# prob_009

## question

光明未来玩具公司计划生产并销售机器人、模型汽车、积木套装和玩偶。每售出一个机器人可获得 15 美元利润，每售出一个模型汽车可获得 8 美元利润，每售出一套积木可获得 12 美元利润，每售出一个玩偶可获得 5 美元利润。光明未来玩具公司应生产多少种类的玩具以使利润最大化？

现有 1200 单位的塑料。每个机器人需要 30 单位塑料，每个模型汽车需要 10 单位塑料，每套积木需要 20 单位塑料，每个玩偶需要 15 单位塑料。

现有 800 单位的电子元件。每个机器人需要 8 单位电子元件，每个模型汽车需要 5 单位电子元件，每套积木需要 3 单位电子元件，每个玩偶需要 2 单位电子元件。

如果光明未来玩具公司生产机器人，则不生产玩偶。

但是，如果他们生产模型汽车，则也会生产积木。

生产的玩偶数量不能超过生产的模型汽车数量。

### Other Details
- **description**: Bright Future Toys wants to build and sell robots, model cars, building blocks, and dolls. The profit for each robot sold is $15, for each model car sold is $8, for each set of building blocks sold is $12, and for each doll sold is $5. How many types of toys should Bright Future Toys manufacture to maximize profit?
There are 1200 units of plastic available. Each robot requires 30 units of plastic, each model car requires 10 units of plastic, each set of building blocks requires 20 units of plastic, and each doll requires 15 units of plastic.

There are 800 units of electronic components available. Each robot requires 8 units of electronic components, each model car requires 5 units of electronic components, each set of building blocks requires 3 units of electronic components, and each doll requires 2 units of electronic components.

If Bright Future Toys manufactures robots, they will not manufacture dolls.

However, if they manufacture model cars, they will also manufacture building blocks.

The number of dolls manufactured cannot exceed the number of model cars manufactured.
- **ground_truth**: 956
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_019
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_019\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_019\code.py

---

# prob_010

## question

著名的运筹学问题——旅行商问题（Traveling Salesman Problem, TSP）可描述如下：一名旅行推销员从某一城市出发，前往另外两个城市推销商品，并且在返回最初出发城市之前，必须使每个城市恰好被访问一次。各城市之间的距离如下面的表格所示。

| 城市 |   1   |   2   |   3   |   4   |
| ---- | ----- | ----- | ----- | ----- |
| 1    | 0     | 10    | 20    | 12    |
| 2    | 10    | 0     | 5     | 10    |
| 3    | 20    | 5     | 0     | 8     |
| 4    | 15    | 12    | 8     | 0     |

推销员应选择怎样的路线才能使总行程距离最短？尝试为该问题建立一个整数规划模型。

### Other Details
- **description**: The famous Traveling Salesman Problem (TSP) in operations research can be described as follows: A traveling salesman departs from a certain city, visits two other cities to sell merchandise, and must visit each city exactly once before returning to the original starting city. The distances between the cities are provided in the table below.
| City |    1    |    2    |    3    |    4    |
| ---- | ------ | ------ | ------ | ------ |
| 1    | 0    | 10   | 20   | 12   |
| 2    | 10   | 0    | 5    | 10   |
| 3    | 20   | 5    | 0    | 8    |
| 4    | 15   | 12   | 8    | 0    |

What route should the salesman choose to travel in order to minimize the total distance? Try to formulate an integer programming model for this problem.
- **ground_truth**: 35.0
- **problem_type**: MILP
- **problem_size**: Small
- **index**: IndustryOR_prob_069
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_069\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_069\code.py

---

# prob_011

## question

一家商店想要清理上个季节剩余的200件衬衫和100条裤子。商店决定推出两种促销组合：A和B。组合A包含一件衬衫和两条裤子，定价为30英镑；组合B包含三件衬衫和一条裤子，定价为50英镑。商店不希望销售少于20份A组合和10份B组合。应当销售多少份每种组合，才能使此次促销的收入最大化？

尝试为这一问题建立一个模型。

### Other Details
- **description**: A store wants to clear out 200 shirts and 100 pairs of pants from last season. They decide to introduce two promotional packages, A and B. Package A includes one shirt and two pairs of pants, priced at £30. Package B includes three shirts and one pair of pants, priced at £50. The store does not want to sell fewer than 20 A packages and 10 B packages. How many of each package do they need to sell to maximize the revenue from the promotion?

Try to establish a model for this problem.
- **ground_truth**: 3600.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_089
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_089\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_089\code.py

---

# prob_012

## question

一家糖果厂使用原料 A、B、C 加工生产三种不同牌号的糖果 A、B、C。已知各牌号糖果中 A、B、C 三种原料的含量、原料成本、每月各原料的供应上限，以及三种牌号糖果的单位加工费和销售单价，如表 1-7 所示。

表 1-7

| 项目                    | A               | B               | C               | 原料成本（元/千克） | 每月供应上限（千克） |
|:------------------------|:---------------|:---------------|:---------------|:--------------------|:----------------------|
| A                       | ≥ 60%          | ≥ 15%          |                | 2.00               | 2000                  |
| B                       |                |                |                | 1.50               | 2500                  |
| C                       | ≤ 20%          | ≤ 60%          | ≤ 50%          | 1.00               | 1200                  |
| 加工费（元/千克）       | 0.50           | 0.40           | 0.30           |                     |                        |
| 销售单价（元/千克）     | 3.40           | 2.85           | 2.25           |                     |                        |

问：该厂每月应生产三种牌号糖果各多少千克，才能使利润最大？

### Other Details
- **description**: A candy factory uses raw materials A, B, and C to process three different brands of candies, A, B, and C. It is known that the content of A, B, and C in each brand of candy, the cost of raw materials, the monthly limit of each raw material, and the unit processing fee and selling price of the three brands of candies are shown in Table 1-7.

Table 1-7

| Item            | A               | B               | C               | Raw Material Cost (Yuan/kg) | Monthly Limit (kg) |
|:----------------|:---------------|:---------------|:---------------|:-----------------------------|:-------------------|
| A               | ≥ 60%          | ≥ 15%          |                | 2.00                        | 2000               |
| B               |                |                |                | 1.50                        | 2500               |
| C               | ≤ 20%          | ≤ 60%          | ≤ 50%          | 1.00                        | 1200               |
| Processing Fee (Yuan/kg) | 0.50         | 0.40           | 0.30           |                             |                     |
| Selling Price (Yuan/kg)   | 3.40         | 2.85           | 2.25           |                             |                     |

How many kilograms of each of the three brands of candies should the factory produce each month to maximize the profit?
- **ground_truth**: 6160.0
- **problem_type**: LP
- **problem_size**: Small
- **index**: IndustryOR_prob_058
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_058\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_058\code.py

---

# prob_013

## question

一家家具店可以从三家不同的制造商 A、B 和 C 订购椅子。向制造商 A 订购每把椅子的成本是 50 美元，向制造商 B 订购每把椅子的成本是 45 美元，向制造商 C 订购每把椅子的成本是 40 美元。商店需要使总订购成本最小化。

另外，每次向制造商 A 的订购将包含 15 把椅子，而每次向制造商 B 和 C 的订购将包含 10 把椅子。订购的批次数必须是整数。商店需要至少订购 100 把椅子。

每次向制造商 A 的订购将包含 15 把椅子，而每次向制造商 B 和 C 的订购将包含 10 把椅子。商店需要至多订购 500 把椅子。

如果商店决定向制造商 A 订购椅子，那么它还必须至少向制造商 B 订购 10 把椅子。

此外，如果商店决定向制造商 B 订购椅子，则它也必须向制造商 C 订购椅子。

### Other Details
- **description**: A furniture store can choose to order chairs from three different manufacturers: A, B, and C. The cost of ordering each chair from manufacturer A is $50, from manufacturer B is $45, and from manufacturer C is $40. The store needs to minimize the total cost of the order.

Additionally, each order from manufacturer A will include 15 chairs, while each order from manufacturers B and C will include 10 chairs. The number of orders must be an integer. The store needs to order at least 100 chairs.

Each order from manufacturer A will include 15 chairs, while each order from manufacturers B and C will include 10 chairs. The store needs to order at most 500 chairs.

If the store decides to order chairs from manufacturer A, it must also order at least 10 chairs from manufacturer B.

Furthermore, if the store decides to order chairs from manufacturer B, it must also order chairs from manufacturer C.
- **ground_truth**: 4000
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_018
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_018\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_018\code.py

---

# prob_014

## question

一位农民需要将 1000 单位的新鲜农产品从农场运送到附近的市场。农民有三种运输方式可选：马、自行车和手推车。由于自行车和手推车都非常费力，农民希望在这两种运输方式中只选择一种使用。马每次运输产生 80 单位的污染，自行车每次运输产生 0 单位的污染，手推车每次运输产生 0 单位的污染。所有运输过程中产生的污染总量不得超过 1000 单位。使用马进行的运输次数至少为 8 次。马、自行车和手推车每次分别可以运送 55 单位、30 单位和 40 单位的农产品。农民需要确保运送的农产品总量至少为 1000 单位。

### Other Details
- **description**: A farmer needs to transport 1000 units of fresh produce from the farm to a nearby market. The farmer has three transportation options: a horse, a bicycle, and a handcart. Since both the bicycle and handcart are very physically demanding, the farmer wants to choose only one of these two transportation methods. The horse generates 80 units of pollution per trip, the bicycle generates 0 units of pollution, and the handcart generates 0 units of pollution. The total amount of pollution generated by all trips must not exceed 1000 units. At least 8 trips must be made using the horse. The horse, bicycle, and handcart can carry 55 units, 30 units, and 40 units of produce per trip respectively. The farmer needs to ensure that the total amount of transported produce is at least 1000 units.
- **ground_truth**: 640.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_062
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_062\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_062\code.py

---

# prob_015

## question

一家商店为某种商品制订了 7 月至 12 月的进货与销售计划。已知仓库容量不得超过 500 件，且 6 月底库存为 200 件。此后，每个月初进行进货。假设该商品在各月的进货价和销售价如表 1-21 所示。应在每个月进货和销售多少件，才能使总收益最大？

表 1-21  
| 月份 | 7  | 8  | 9  | 10 | 11 | 12 |
|------|----|----|----|----|----|----|
| 进货价 | 28 | 24 | 25 | 27 | 23 | 23 |
| 销售价 | 29 | 24 | 26 | 28 | 22 | 25 |

### Other Details
- **description**: A store has formulated a purchase and sales plan for a certain product from July to December. It is known that the warehouse capacity must not exceed 500 units, with 200 units in stock at the end of June. Thereafter, purchases are made at the beginning of each month. Assume the purchase and selling prices of this product for each month are shown in Table 1-21. How much should be purchased and sold each month to maximize the total revenue?

Table 1-21
| Month | 7  | 8  | 9  | 10 | 11 | 12 |
|-------|----|----|----|----|----|----|
| Buy   | 28 | 24 | 25 | 27 | 23 | 23 |
| Sell  | 29 | 24 | 26 | 28 | 22 | 25 |
- **ground_truth**: 9100.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_049
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_049\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_049\code.py

---

# prob_016

## question

一家便利超市计划在本市西北郊新建住宅区内开设若干连锁店。为方便居民购物，从任何一个住宅区到某一连锁店的距离不得超过 $800 \mathrm{~m}$。表 5-1 给出了各个新建住宅区及其半径 $800 \mathrm{~m}$ 范围内所包含的住宅区。问题：超市最少需要在这些住宅区中建设多少个连锁店？应选择在哪些住宅区建设？

| 区域代号 | $800 \mathrm{~m}$ 半径内包含的住宅区 |
|----------|----------------------------------------|
| A        | A, C, E, G, H, I                      |
| B        | B, H, I                               |
| C        | A, C, G, H, I                         |
| D        | D, J                                  |
| E        | A, E, G                               |
| F        | F, J, K                               |
| G        | A, C, E, G                            |
| H        | A, B, C, H, I                         |
| I        | A, B, C, H, I                         |
| J        | D, F, J, K, L                         |
| K        | F, J, K, L                            |
| L        | J, K, L                               |

### Other Details
- **description**: A convenience supermarket is planning to open several chain stores in a newly built residential area in the northwest suburb of the city. For shopping convenience, the distance from any residential area to one of the chain stores should not exceed $800 \mathrm{~m}$. Table 5-1 shows the new residential areas and the residential areas within a radius of $800 \mathrm{~m}$ from each of them. Question: What is the minimum number of chain stores the supermarket needs to build among the mentioned residential areas, and in which residential areas should they be built?

| Area Code | Residential Areas within $800 \mathrm{~m}$ Radius |
|-----------|---------------------------------------------------|
| A         | A, C, E, G, H, I                                  |
| B         | B, H, I                                           |
| C         | A, C, G, H, I                                     |
| D         | D, J                                              |
| E         | A, E, G                                           |
| F         | F, J, K                                           |
| G         | A, C, E, G                                        |
| H         | A, B, C, H, I                                     |
| I         | A, B, C, H, I                                     |
| J         | D, F, J, K, L                                     |
| K         | F, J, K, L                                        |
| L         | J, K, L                                           |
- **ground_truth**: 3
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_009
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_009\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_009\code.py

---

# prob_017

## question

Haus Toys 可以生产和销售玩具卡车、玩具飞机、玩具船和玩具火车。每售出一辆卡车的利润为 5 美元，每架飞机为 10 美元，每艘船为 8 美元，每列火车为 7 美元。Haus Toys 应该生产多少种玩具以使利润最大化？

共有 890 个单位的木材可用。每辆卡车需要 12 个单位木材，每架飞机需要 20 个单位，每艘船需要 15 个单位，每列火车需要 10 个单位。

共有 500 个单位的钢材可用。每架飞机需要 3 个单位钢材，每艘船需要 5 个单位，每列火车需要 4 个单位，每辆卡车需要 6 个单位。

如果 Haus Toys 生产卡车，则不生产火车。

但是，如果他们生产船，则也会生产飞机。

生产的玩具船数量不能超过生产的玩具火车数量。

### Other Details
- **description**: Haus Toys can manufacture and sell toy trucks, toy airplanes, toy boats, and toy trains. The profit for each truck sold is $5, each airplane $10, each boat $8, and each train $7. How many types of toys should Haus Toys manufacture to maximize profits?

There are 890 units of wood available. Each truck requires 12 units, each airplane 20 units, each boat 15 units, and each train 10 units.

There are 500 units of steel available. Each airplane requires 3 units, each boat 5 units, each train 4 units, and each truck 6 units.

If Haus Toys manufactures trucks, they will not manufacture trains.

However, if they manufacture boats, they will also manufacture airplanes.

The number of toy boats manufactured cannot exceed the number of toy trains manufactured.
- **ground_truth**: 623
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_008
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_008\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_008\code.py

---

# prob_018

## question

一家公司生产两种产品（X 和 Y）。生产 X 和 Y 所需的资源分为两个部分：用于自动加工的机器时间和用于手工精加工的工匠时间。下表给出了每种产品所需的分钟数：

| 项目 | 机器时间（分钟） | 工匠时间（分钟） |
| :---: | :---: | :---: |
| X | 13 | 20 |
| Y | 19 | 29 |

公司在下一个工作周可用的机器时间为 40 小时，但工匠时间只有 35 小时。机器时间的成本为每小时 10 英镑，工匠时间的成本为每小时 2 英镑。机器和工匠的空闲时间不产生任何成本。对于每一件生产出来的产品（所有生产出来的产品都可以售出），产品 X 的收入为 20 英镑，产品 Y 的收入为 30 英镑。公司有一项特定合同，要求每周必须为某客户生产 10 件产品 X。为该问题建立一个模型。

### Other Details
- **description**: A company is producing two products (X and Y). The resources required for the production of X and Y are divided into two parts: machine time for automated processing and craftsman time for manual finishing. The table below shows the number of minutes required for each product:

| Item | Machine Time (minutes) | Craftsman Time (minutes) |
| :---: | :---: | :---: |
| X | 13 | 20 |
| Y | 19 | 29 |

The company has 40 hours of machine time available in the next working week, but only 35 hours of craftsman time. The cost of machine time is £10 per hour, and the cost of craftsman time is £2 per hour. Idle time for machines and craftsmen incurs no cost. For each product produced (all products produced will be sold), the revenue for product X is £20, and the revenue for product Y is £30. The company has a specific contract that requires 10 units of product X to be produced for a customer each week. Formulate a model for this problem.
- **ground_truth**: 1861.466666667
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_084
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_084\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_084\code.py

---

# prob_019

## question

某人拥有资金 300,000 元，在今后三年中有如下投资项目可供选择：  
(1) 在三年内的每年年初都可以进行投资，年收益率为投资额的 20%，本息可在下一年继续用于投资；  
(2) 只允许在第一年年初投资，并在第二年年末收回，本息合计为投资额的 150%，但投资额上限为 150,000 元；  
(3) 允许在第二年年初进行投资，并在第三年年末收回，本息合计为投资额的 160%，且投资额上限为 200,000 元；  
(4) 允许在第三年年初进行投资，并在一年后收回，收益率为 40%，投资额上限为 100,000 元。  

第一章  线性规划与单纯形法  

试为该人确定一项投资计划，使第三年年末所得到的本利和最大。

### Other Details
- **description**: Someone has a fund of 300,000 yuan and has the following investment projects in the next three years:
(1) Investment can be made at the beginning of each year within three years, with an annual profit of 20% of the investment amount, and the principal and interest can be used for investment in the following year;
(2) Investment is only allowed at the beginning of the first year, and it can be recovered at the end of the second year, with the total principal and interest amounting to 150% of the investment amount, but the investment limit is no more than 150,000 yuan;
(3) Investment is allowed at the beginning of the second year within three years, and it can be recovered at the end of the third year, with the total principal and interest amounting to 160% of the investment amount, and the investment limit is 200,000 yuan;
(4) Investment is allowed at the beginning of the third year within three years, and it can be recovered in one year with a profit of 40%, and the investment limit is 100,000 yuan.
Chapter One: Linear Programming and Simplex Method
Try to determine an investment plan for this person that maximizes the principal and interest at the end of the third year.
- **ground_truth**: 580000
- **problem_type**: LP
- **problem_size**: Small
- **index**: IndustryOR_prob_027
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_027\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_027\code.py

---

# prob_020

## question

一所学校正在为400名学生准备一次旅行。运输公司拥有10辆每辆有50个座位的大巴士和8辆每辆有40个座位的中巴士，但只有9名可用司机。一辆大巴士的租金是800英镑，一辆中巴士的租金是600英镑。计算应使用每种类型的巴士各多少辆以达到最低成本。

尝试为这个问题建立一个模型。

### Other Details
- **description**: A school is preparing a trip for 400 students. The transportation company has 10 buses with 50 seats each and 8 minibuses with 40 seats each, but only 9 drivers are available. The rental cost for a bus is £800, and the rental cost for a minibus is £600. Calculate how many of each type of bus should be used to achieve the lowest cost.

Try to formulate a model for this problem.
- **ground_truth**: 6200.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_091
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_091\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_091\code.py

---

# prob_021

## question

某大学的一名运筹学专业硕士研究生被要求从以下七门课程中选修：高等数学、运筹学、数据结构、管理统计、计算机模拟、计算机程序设计以及预测，其目标是：数学类课程选两门，运筹学类课程选两门，计算机科学类课程选两门。

有些课程只属于一个类别：高等数学属于数学类，计算机程序设计属于计算机科学类。  
但也有一些课程同时属于多个类别：运筹学同时属于运筹学类和数学类，数据结构同时属于计算机科学类和数学类，管理统计同时属于数学类和运筹学类，计算机模拟同时属于计算机科学类和运筹学类，预测同时属于运筹学类和数学类。  
属于多个类别的课程可以同时满足其所对应的多个类别的选课要求。

此外，一些课程有先修课要求：  
选修计算机模拟或数据结构之前必须先修计算机程序设计；  
选修管理统计之前必须先修高等数学；  
选修预测之前必须先修管理统计。

问题是：为了满足上述要求，该硕士研究生最少需要修读多少门课程？具体应选修哪些课程？

### Other Details
- **description**: A master's student in Operations Research at a certain university is required to select two courses in mathematics, two in operations research, and two in computer science from a total of seven courses: Calculus, Operations Research, Data Structures, Management Statistics, Computer Simulation, Computer Programming, and Forecasting. Some courses belong to only one category: Calculus falls under Mathematics, Computer Programming under Computer Science. However, some courses fall under multiple categories: Operations Research can be considered both Operations Research and Mathematics, Data Structures both Computer Science and Mathematics, Management Statistics both Mathematics and Operations Research, Computer Simulation both Computer Science and Operations Research, and Forecasting both Operations Research and Mathematics. Courses that fall under multiple categories can fulfill the requirement of both categories simultaneously. Additionally, some courses have prerequisites: Computer Simulation or Data Structures requires Computer Programming first, Management Statistics requires Calculus first, and Forecasting requires Management Statistics first. The question is: What is the minimum number of courses a master's student must take, and which specific courses, to meet the above requirements?
- **ground_truth**: 4.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_054
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_054\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_054\code.py

---

# prob_022

## question

一家公司需要决定是否从五名候选人中雇用若干人加入其研发团队。候选人 F、G、H、I 和 J 的薪资要求分别为 12,000 美元、15,000 美元、18,000 美元、5,000 美元和 10,000 美元。公司希望在不超出预算的前提下，使支付给候选人的总金额最小化。

公司的预算为 40,000 美元，并且他们希望最多雇用 4 名新员工。

候选人的技能等级如下：
候选人 F：等级 2  
候选人 G：等级 3  
候选人 H：等级 4  
候选人 I：等级 1  
候选人 J：等级 2  

公司需要确保被雇用员工的总技能等级至少为 8。

各候选人的项目管理经验年限如下：
候选人 F：1 年  
候选人 G：2 年  
候选人 H：2 年  
候选人 I：5 年  
候选人 J：4 年  

他们希望团队的项目管理经验总年限至少为 8 年。

由于候选人 G 和 J 的技术背景相似，公司至多只能选择其中一人。

### Other Details
- **description**: A company needs to decide whether to hire some of the five candidates to join their R&D team. The salary requirements for candidates F, G, H, I, and J are $12,000, $15,000, $18,000, $5,000, and $10,000 respectively. The company wants to minimize the total amount paid to candidates without exceeding the budget.

The company's budget is $40,000 and they wish to hire a maximum of 4 new employees.

The skill levels of the candidates are as follows:
Candidate F: Level 2
Candidate G: Level 3
Candidate H: Level 4
Candidate I: Level 1
Candidate J: Level 2

The company needs to ensure that the total skill level of the hired employees is at least 8.

The project management experience years of each candidate are as follows:
Candidate F: 1 year
Candidate G: 2 years
Candidate H: 2 years
Candidate I: 5 years
Candidate J: 4 years

They hope the total project management experience of the team is at least 8 years.

Due to the similar technical background of candidates G and J, the company can choose at most one of them.
- **ground_truth**: 38000.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_063
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_063\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_063\code.py

---

# prob_023

## question

一家木材仓储与运输公司拥有一座大型仓库，用于存储和运输待售木材。由于木材价格具有季节性波动，该公司在每个季度初购入木材，其中一部分在本季度内销售，另一部分存入仓库以备今后销售。已知该公司仓库的最大库存容量为 200000 m³，且库存费用为 $(a+b u)$ 元/m³，其中 $a=70$，$b=100$，$u$ 为库存时间（以季度为单位）。各季度的进货价格、销售价格和预计最大销售量如表 1-18 所示。

表 1-18  
| 季节 | 进货价格 (万元/万 m²) | 销售价格 (万元/万 m²) | 预计最大销售量 (万 m³) |
|------|-----------------------|------------------------|--------------------------|
| 冬季 | 410                   | 425                    | 100                      |
| 春季 | 430                   | 440                    | 140                      |
| 夏季 | 460                   | 465                    | 200                      |
| 秋季 | 450                   | 455                    | 160                      |

由于木材不宜长期存放，要求在秋季末将库存全部售完。试为该问题建立一个线性规划模型，以使公司的年利润最大。

### Other Details
- **description**: A timber storage and transport company has a large warehouse for storing and transporting timber for sale. Due to seasonal price fluctuations, the company purchases timber at the beginning of each quarter, with part of it being sold within the quarter and part being stored for future sales. It is known that the maximum storage capacity of the company’s warehouse is 200,000 m³, and the storage cost is $(a+b u)$ yuan/m³, where $a=70$, $b=100$, and $u$ is the storage time (in quarters). The purchase and sale prices for each quarter and the estimated maximum sales volumes are shown in Table 1-18.

Table 1-18
| Quarter | Purchase Price (10,000 yuan/10,000 m²) | Sale Price (10,000 yuan/10,000 m²) | Estimated Maximum Sales Volume (10,000 m³) |
|---------|----------------------------------------|------------------------------------|---------------------------------------------|
| Winter  | 410                                    | 425                                | 100                                         |
| Spring  | 430                                    | 440                                | 140                                         |
| Summer  | 460                                    | 465                                | 200                                         |
| Autumn  | 450                                    | 455                                | 160                                         |

Since timber is not suitable for long-term storage, all inventory should be sold by the end of autumn. Try to establish a linear programming model for this problem to maximize the company's annual profit.
- **ground_truth**: 4700.0
- **problem_type**: LP
- **problem_size**: Small
- **index**: IndustryOR_prob_075
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_075\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_075\code.py

---

# prob_024

## question

一种产品可以在四台设备中的任意一台上加工：A、B、C 或 D。当每台设备投入使用时，其准备完成费用、该产品的单位生产成本以及每台设备的最大加工能力如表 5-7 所示。若需要生产 2000 件该产品，如何使总成本最小？试建立数学模型。

表 5-7  
| 设备 | 准备完成费用（元） | 单位生产成本（元/件） | 最大加工能力（件） |
|------|--------------------|------------------------|--------------------|
| A    | 1000               | 20                     | 900                |
| B    | 920                | 24                     | 1000               |
| C    | 800                | 16                     | 1200               |
| D    | 700                | 28                     | 1600               |

### Other Details
- **description**: A product can be processed on any one of the four devices: A, B, C, or D. The preparation completion costs when each device is enabled, the unit production cost for the product, and the maximum processing capacity of each device are shown in Table 5-7. If 2000 units of the product need to be produced, how can the total cost be minimized? Try to establish a mathematical model.

Table 5-7
| Device | Prep Completion Cost (Yuan) | Unit Production Cost (Yuan/Unit) | Maximum Processing Capacity (Units) |
|--------|------------------------------|----------------------------------|------------------------------------|
| A      | 1000                         | 20                               | 900                                |
| B      | 920                          | 24                               | 1000                               |
| C      | 800                          | 16                               | 1200                               |
| D      | 700                          | 28                               | 1600                               |
- **ground_truth**: 37000.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_060
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_060\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_060\code.py

---

# prob_025

## question

某工厂生产三种产品：I、II 和 III。每种产品都必须经过两个加工工序 A 和 B。工厂有两台可进行工序 A 的设备，记为 A1 和 A2；有三台可进行工序 B 的设备，记为 B1、B2 和 B3。产品 I 在工序 A 和 B 上可以在任意设备上加工；产品 II 在工序 A 上可以在任意 A 类设备上加工，但在工序 B 上只能在 B1 上加工；产品 III 只能在 A2 和 B2 上加工。已知各产品在不同设备上的单位加工时间、原材料成本、产品销售单价、设备的有效工作小时数以及设备满负荷运转的成本（见表 1-4），要求安排最优生产计划，使工厂利润最大。

表 1-4  
| 设备                          | 产品 I | 产品 II | 产品 III | 有效工作小时数 | 满负荷运转成本（元） |
|-------------------------------|--------|---------|----------|----------------|----------------------|
| A1                            | 5      | 10      |          | 6000           | 300                  |
| A2                            | 7      | 9       | 12       | 10000          | 321                  |
| B1                            | 6      | 8       |          | 4000           | 250                  |
| B2                            | 4      |         | 11       | 7000           | 783                  |
| B3                            | 7      |         |          | 4000           | 200                  |
| 原材料成本（元/件）          | 0.25   | 0.35    | 0.50     |                |                      |
| 单价（元/件）                | 1.25   | 2.00    | 2.80     |                |                      |

### Other Details
- **description**: A factory produces three types of products: I, II, and III. Each product needs to go through two processing procedures, A and B. The factory has two pieces of equipment that can complete process A, denoted as A1 and A2; it has three pieces of equipment that complete process B, denoted as B1, B2, and B3. Product I can be processed on any equipment for A and B; Product II can be processed on any A equipment but only on B1 for process B; Product III can only be processed on A2 and B2. Given the unit processing time on various machines, raw material costs, product sale prices, effective machine hours, and the costs of operating the machines at full capacity as shown in Table 1-4, the task is to arrange the optimal production plan to maximize the factory's profit.

Table 1-4
| Equipment  | Product I | Product II | Product III | Effective Machine Hours | Operating Costs at Full Capacity (Yuan) |
|------------|-----------|------------|-------------|--------------------------|------------------------------------------|
| A1         | 5         | 10         |             | 6000                     | 300                                      |
| A2         | 7         | 9          | 12          | 10000                    | 321                                      |
| B1         | 6         | 8          |             | 4000                     | 250                                      |
| B2         | 4         |            | 11          | 7000                     | 783                                      |
| B3         | 7         |            |             | 4000                     | 200                                      |
| Raw Material Cost (Yuan/Unit) | 0.25 | 0.35       | 0.50       |                          |                                          |
| Unit Price (Yuan/Unit)        | 1.25 | 2.00       | 2.80       |                          |                                          |
- **ground_truth**: 1146.57
- **problem_type**: LP
- **problem_size**: Small
- **index**: IndustryOR_prob_026
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_026\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_026\code.py

---

# prob_026

## question

Healthy Pet Foods 公司生产两种狗粮产品：Meaties 和 Yummies。每包 Meaties 含有 2 磅谷物和 3 磅肉；每包 Yummies 含有 3 磅谷物和 1.5 磅肉。公司认为，只要能够生产出来，就可以卖出任意数量的狗粮。Meaties 的售价为每包 2.80 美元，Yummies 的售价为每包 2.00 美元。公司的生产受到若干约束条件的限制。首先，每月最多可购买 400,000 磅谷物，谷物的价格为每磅 0.20 美元；每月最多可购买 300,000 磅肉，肉的价格为每磅 0.50 美元。另外，生产 Meaties 需要一台专用机器，其月产能为 90,000 包。搅拌和包装狗粮的变动成本为：Meaties 每包 0.25 美元，Yummies 每包 0.20 美元。详细信息见表 B-1。

**表 B-1 Healthy Pet Foods 数据**

|                    | Meaties            | Yummies      |
|--------------------|--------------------|--------------|
| 每包售价           | $2.80              | $2.00        |
| 原材料             |                    |              |
| - 谷物             | 2.0 磅             | 3.0 磅       |
| - 肉               | 3.0 磅             | 1.5 磅       |
| 变动成本           | $0.25/包           | $0.20/包     |
| 资源               |                    |              |
| Meaties 产能       | 90,000 包/月       |              |
| 每月可用谷物       | 400,000 磅         |              |
| 每月可用肉         | 300,000 磅         |              |

假设你是 Healthy Pet Foods 公司狗粮部门的经理。你的薪酬取决于该部门的利润，因此你将努力使利润最大化。你应当如何运营该部门，才能同时最大化部门利润和你的薪酬？

### Other Details
- **description**: Healthy Pet Foods Company produces two types of dog food: Meaties and Yummies. Each pack of Meaties contains 2 pounds of grains and 3 pounds of meat; each pack of Yummies contains 3 pounds of grains and 1.5 pounds of meat. The company believes it can sell any quantity of dog food that it can produce. Meaties sell for $2.80 per pack, and Yummies sell for $2.00 per pack. The company's production is subject to several constraints. First, a maximum of 400,000 pounds of grains can be purchased each month at a price of $0.20 per pound of grains. A maximum of 300,000 pounds of meat can be purchased each month at a price of $0.50 per pound of meat. Additionally, a special machine is required to produce Meaties, with a monthly capacity of 90,000 packs. The variable costs for mixing and packaging dog food are $0.25 per pack (Meaties) and $0.20 per pack (Yummies). Detailed information is provided in Table B-1.

**Table B-1 Healthy Pet Foods Data**

|                    | Meaties      | Yummies    |
|--------------------|--------------|------------|
| Price per pack     | $2.80        | $2.00      |
| Raw materials      |              |            |
| - Grains           | 2.0 lbs      | 3.0 lbs    |
| - Meat             | 3.0 lbs      | 1.5 lbs    |
| Variable cost      | $0.25/pack   | $0.20/pack |
| Resources          |              |            |
| Meaties capacity   | 90,000 packs/month |       |
| Monthly available grains | 400,000 lbs |      |
| Monthly available meat | 300,000 lbs |        |

Assume you are the manager of the dog food department at Healthy Pet Foods Company. Your salary is based on the department's profit, so you will try to maximize profit. How should you operate the department to maximize both the profit and your salary?
- **ground_truth**: 77500.0
- **problem_type**: LP
- **problem_size**: Toy
- **index**: IndustryOR_prob_085
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_085\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_085\code.py

---

# prob_027

## question

一家工厂生产两种微型计算机型号，A 型和 B 型。每种型号都需要经过相同的两个工序。各型号在每个工序上的加工时间、销售利润以及工厂每周最大加工能力如表 3.1 所示。

表 3.1

| 工序 | 型号 | | 每周最大加工能力 |
| :---: | :---: | :---: | :---: |
| | $A$ | $B$ | |
| I（小时/台） | 4 | 6 | 150 |
| II（小时/台） | 3 | 2 | 70 |
| 利润（元/台） | 300 | 450 | |

给定工厂的经营目标：

$p_{1}$：每周总利润不得低于 10 000 元；

$p_{2}$：由于合同要求，每周至少生产 10 台 A 型产品，且至少生产 15 台 B 型产品；

$p_{3}$：I 工序的加工时间每周必须恰好为 150 小时，II 工序的加工时间最好能被充分利用，并且可以适当加班；

$p_{4}$：若在 II 工序中安排加班生产，则 A 型和 B 型产品的单位利润分别减少 20 元和 25 元，且 II 工序每周最多允许加班 30 小时。建立该问题的数学模型。

### Other Details
- **description**: A factory produces two models of microcomputers, A and B. Each model requires the same two processes. The processing time, sales profit, and the factory’s maximum weekly processing capacity for each model are shown in Table 3.1.

Table 3.1

| Process | Model | | Maximum Weekly Processing Capacity |
| :---: | :---: | :---: | :---: |
| | $A$ | $B$ | |
| I (hours/unit) | 4 | 6 | 150 |
| II (hours/unit) | 3 | 2 | 70 |
| Profit (yuan/unit) | 300 | 450 | |

Given the factory's business goals:

$p_{1}$: The total weekly profit should not be less than 10,000 yuan;

$p_{2}$: Due to contract requirements, at least 10 units of model A and at least 15 units of model B must be produced each week;

$p_{3}$: The processing time for Process I should be exactly 150 hours per week, and the processing time for Process II should ideally be fully utilized, with potential for appropriate overtime;

$p_{4}$: If products are produced during overtime in Process II, the profit per unit is reduced by 20 yuan for model A and 25 yuan for model B, and the maximum overtime for Process II is 30 hours per week. Formulate the mathematical model for this problem.
- **ground_truth**: 11250.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_047
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_047\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_047\code.py

---

# prob_028

## question

李家有 5 个孩子：Alice、Bob、Charlie、Diana 和 Ella。带 Alice 的费用是 1000 美元，Bob 是 900 美元，Charlie 是 600 美元，Diana 是 500 美元，Ella 是 700 美元。夫妇应带哪些孩子才能使带孩子的总费用最小？

他们在这次旅行中最多可以带 3 个孩子。

Bob 是最小的，所以李家一定会带他。

如果夫妇带 Alice，就不会带 Diana，因为 Alice 和她合不来。

如果夫妇带 Bob，就不会带 Charlie，因为 Bob 和他合不来。

如果带 Charlie，就必须同时带 Diana。

如果带 Diana，就必须同时带 Ella。

尽管费用不低，李家还是决定至少带两个孩子。

### Other Details
- **description**: The Li family has 5 children: Alice, Bob, Charlie, Diana, and Ella. The cost to take Alice is $1000, Bob is $900, Charlie is $600, Diana is $500, and Ella is $700. Which children should the couple take to minimize the total cost of taking the children?

They can take up to 3 children on the upcoming trip.

Bob is the youngest, so the Li family will definitely take him.

If the couple takes Alice, they will not take Diana because Alice does not get along with her.

If the couple takes Bob, they will not take Charlie because Bob does not get along with him.

If they take Charlie, they must also take Diana.

If they take Diana, they must also take Ella.

Despite the cost, the Li family has decided to take at least two children.
- **ground_truth**: 1600.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: IndustryOR_prob_072
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_072\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_072\code.py

---

# prob_029

## question

在某医院，24 小时内各时间段所需的护士人数如下：2:00–6:00 需要 10 人，6:00–10:00 需要 15 人，10:00–14:00 需要 25 人，14:00–18:00 需要 20 人，18:00–22:00 需要 18 人，22:00–2:00 需要 12 人。护士排班开始时间共有 6 批，分别在 2:00、6:00、10:00、14:00、18:00 和 22:00，每位护士连续工作 8 小时。请确定：如果医院可以聘用与正式护士工作时间相同的合同护士，且正式护士的报酬为 10 元/小时，合同护士的报酬为 15 元/小时，医院是否应当聘用合同护士？如果应当聘用，应聘用多少人？

### Other Details
- **description**: The number of nurses required in each time period over 24 hours at a certain hospital is as follows: 2:00-6:00 - 10 people, 6:00-10:00 - 15 people, 10:00-14:00 - 25 people, 14:00-18:00 - 20 people, 18:00-22:00 - 18 people, 22:00-2:00 - 12 people. Nurses start shifts in 6 batches at 2:00, 6:00, 10:00, 14:00, 18:00, and 22:00 and work continuously for 8 hours. Please determine: If the hospital can hire contract nurses with the same working hours as regular nurses, and if the pay for regular nurses is 10 yuan/hour and for contract nurses is 15 yuan/hour, should the hospital hire contract nurses and if so, how many?
- **ground_truth**: 4240.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: IndustryOR_prob_050
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_050\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_050\code.py

---

# prob_030

## question

某公司将两种原油（A 和 B）进行调和生产两种汽油（Ⅰ 型和 Ⅱ 型）。在汽油 Ⅰ 型和 Ⅱ 型中，原油 A 的最低配比分别为 50% 和 60%。两种汽油的销售价格分别为 4800 元/吨和 5600 元/吨。公司现有库存为原油 A 500 吨、原油 B 1000 吨，并且还可以从市场上购买最多 1500 吨原油 A。市场上原油 A 的价格为：购买数量在 500 吨以内的部分为 10000 元/吨；超过 500 吨但不超过 1000 吨的部分为 8000 元/吨；超过 1000 吨的部分为 6000 元/吨。问公司应如何安排原油的采购与加工？

### Other Details
- **description**: A company blends two types of crude oil (A and B) to produce two types of gasoline (Type I and Type II). The minimum proportion of crude oil A in gasoline Types I and II is 50% and 60%, respectively. The selling prices are 4800 yuan/t and 5600 yuan/t, respectively. The company has current inventories of 500 t of crude oil A and 1000 t of crude oil B, and they can purchase up to 1500 t of crude oil A from the market. The market price for crude oil A is: 10,000 yuan/t for purchases up to 500 t; 8,000 yuan/t for the portion exceeding 500 t but not exceeding 1000 t; 6,000 yuan/t for the portion exceeding 1000 t. How should the company plan its purchasing and processing of crude oil?
- **ground_truth**: 5000000.0
- **problem_type**: MILP
- **problem_size**: Small
- **index**: IndustryOR_prob_093
- **model_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_093\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\IndustryOR\prob_093\code.py

---

# prob_031

## question

一个机场有两个可用登机口（G1、G2），需要将四个航班分配到这两个登机口上（F1：10:00-11:00，F2：10:30-11:30，F3：11:15-12:30，F4：10:45-11:45）。分配到同一登机口的航班，其使用时间区间不能重叠。目标是最小化航班延误，其中每超过计划起飞时间 1 分钟的延误产生 1 个单位的成本。应如何将这些航班分配到各登机口以实现这一目标？

### Other Details
- **description**: An airport has two available gates (G1, G2) and needs to assign three flights (F1: 10:00-11:00, F2: 10:30-11:30, F3: 11:15-12:30, F4: 10:45-11:45) to these gates. Flights assigned to the same gate must have non-overlapping time intervals, and the goal is to minimize flight delays, where each 1-minute delay beyond the scheduled departure time incurs a cost of 1 unit. How should the flights be assigned to the gates to achieve this objective?
- **ground_truth**: 30.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_002
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_002\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_002\code.py

---

# prob_032

## question

LogiTrans 运营两个配送中心，以管理三种类型货物到各区域枢纽的发运。每个配送中心在同一时间最多可配备七名物流协调员。协调员在配送中心 1 的周薪为 500 美元，在配送中心 2 的周薪为 900 美元。启用一个配送中心一周会产生固定成本：配送中心 1 为 1,000 美元，配送中心 2 为 2,000 美元。在一个配送中心的一周时间内，每名协调员可处理各类型货物的发运数量如表所示：

| 货物类型 \\ 配送中心 | 中心 1 | 中心 2 |
|----------------------|--------|--------|
| 类型 1               | 20     | 25     |
| 类型 2               | 18     | 22     |
| 类型 3               | 15     | 20     |

每周，LogiTrans 必须保证至少处理 120 件类型 1 货物、至少 150 件类型 2 货物以及至少 200 件类型 3 货物。如何在满足每周发运需求的前提下使总成本最小？

### Other Details
- **description**: LogiTrans operates two distribution centers to manage the shipment of three types of goods to various regional hubs. Each distribution center can be staffed by up to seven logistics coordinators at a time. Coordinators are paid $500 per week at distribution center 1 and $900 per week at distribution center 2. Activating a distribution center for a week incurs a fixed cost of $1,000 for center 1 and $2,000 for center 2. During a week at a distribution center, each coordinator can process the number of shipments of each type of goods as shown in Table as follows:

| Goods Type \ Distribution Center | Center 1 | Center 2 |
|----------------------------------|----------|----------|
| Type 1                           | 20       | 25       |
| Type 2                           | 18       | 22       |
| Type 3                           | 15       | 20       |

Each week, LogiTrans must ensure that at least 120 shipments of goods type 1, at least 150 shipments of goods type 2, and at least 200 shipments of goods type 3 are processed. How to minimize the total cost of meeting weekly shipment demands?
- **ground_truth**: 11000.0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: LogiOR_prob_030
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_030\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_030\code.py

---

# prob_033

## question

LogiTech 的主配送中心由两个关键区域组成：一个位于坐标 (0, 0) 的包装区以及一个位于坐标 (800, 600) 的装货码头（距离单位为米）。公司必须确定一个分拣设施和一个存储设施的选址，以使货物在中心内移动的日常成本最小化。配送中心内的所有移动都限制为沿东西向或南北向行进。

各区域之间每日往返次数如下：

| 起点/终点       | 包装区 | 装货码头 | 分拣设施 | 存储设施 |
|-----------------|--------|----------|----------|----------|
| 包装区          | -      | 20       | 15       | 10       |
| 装货码头        | 20     | -        | 25       | 30       |
| 分拣设施        | 15     | 25       | -        | 40       |
| 存储设施        | 10     | 30       | 40       | -        |

货物移动的成本为每米每次 $0.50。分拣设施和存储设施必须位于配送中心边界之内，该中心的范围为从 (0, 0) 到 (1000, 1000)。并且四个区域中任意两个区域之间的距离至少为 100 个单位。

LogiTech 应如何布置分拣设施和存储设施的位置，以使每日运输成本最小化？

### Other Details
- **description**: LogiTech’s main distribution center is organized into two key areas: a packaging area located at coordinates (0, 0) and a loading dock located at coordinates (800, 600) (distances are in meters). The company must determine where to locate a sorting facility and a storage facility to minimize the daily cost of moving goods through the center. All movement within the distribution center is restricted to either east–west or north–south directions.  

The number of daily trips between the areas are as follows:  

| From/To          | Packaging Area | Loading Dock | Sorting Facility | Storage Facility |
|-------------------|----------------|--------------|-------------------|------------------|
| Packaging Area    | -              | 20           | 15                | 10               |
| Loading Dock      | 20             | -            | 25                | 30               |
| Sorting Facility  | 15             | 25           | -                 | 40               |
| Storage Facility  | 10             | 30           | 40                | -                |

The cost of moving goods is $0.50 per meter per trip. The sorting facility and storage facility must be located within the boundaries of the distribution center, which spans from (0, 0) to (1000, 1000). And any two areas among the four areas are at least 100 units apart.

How should LogiTech locate the sorting and storage facilities to minimize daily transportation costs?
- **ground_truth**: 70000.0
- **problem_type**: LP
- **problem_size**: Toy
- **index**: LogiOR_prob_049
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_049\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_049\code.py

---

# prob_034

## question

LogiFlow Solutions 负责两类包裹的配送：标准件（Standard）和快件（Express）。每个包裹必须完全通过空运或完全通过地运来运输。共有 150 立方米的空运容量和 210 立方米的地运容量可用。一个标准件包裹要么需要 17 立方米的空运容量，要么需要 30 立方米的地运容量；一个快件包裹要么需要 5 立方米的空运容量，要么需要 13 立方米的地运容量。每个标准件包裹可获得 40 美元收入，每个快件包裹可获得 15 美元收入。如何最大化 LogiFlow Solutions 在包裹配送中的总收入？

### Other Details
- **description**: LogiFlow Solutions manages the distribution of two types of packages: Standard and Express. Each package must be transported entirely by air freight or entirely by ground shipping. A total of 150 cubic meters of air freight capacity and 210 cubic meters of ground shipping capacity are available. A Standard package requires either 17 cubic meters of air freight or 30 cubic meters of ground shipping, while an Express package requires either 5 cubic meters of air freight or 13 cubic meters of ground shipping. Each Standard package generates $40 in revenue, and each Express package generates $15 in revenue. How to maximize LogiFlow Solutions' total revenue from package distribution?
- **ground_truth**: 730
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: LogiOR_prob_042
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_042\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_042\code.py

---

# prob_035

## question

LogiCorp 运营一个物流网络，运输两种类型的产品：基础型（Basic）和高级型（Advanced）。生产这些产品所需的原材料可以以每单位 5 美元的价格购买。加工 1 单位原材料需要 2 小时仓储劳动力。每单位已加工的原材料可产出 2 单位基础型产品和 3 单位高级型产品。基础型产品可按 10 美元/单位出售，高级型产品可按 12 美元/单位出售。LogiCorp 还可以选择对基础型和高级型产品进行进一步加工，生产高端基础型（Premium Basic）和高端高级型（Premium Advanced），其售价分别为 20 美元/单位和 25 美元/单位。每加工 1 单位基础型产品需要额外 3 小时仓储劳动力和 5 美元加工成本，产出 1 单位高端基础型产品。每加工 1 单位高级型产品需要额外 4 小时仓储劳动力和 6 美元加工成本，产出 1 单位高端高级型产品。每年，LogiCorp 拥有 8,000 小时的仓储劳动力可用，并且最多可以购买 3,000 单位原材料。LogiCorp 应如何最大化其利润？假设仓储劳动力成本为固定成本，且原材料只能以整数单位购买。

### Other Details
- **description**: LogiCorp operates a logistics network that ships two types of products: Basic and Advanced. The raw materials needed to produce these products can be purchased for $5 per unit. Processing 1 unit of raw material requires 2 hours of warehouse labor. Each unit of processed raw material yields 2 units of Basic Product and 3 units of Advanced Product. Basic Product can be sold for $10/unit, and Advanced Product can be sold for $12/unit. LogiCorp also has the option of further processing Basic and Advanced Products to produce Premium Basic and Premium Advanced, which sell for $20/unit and $25/unit, respectively. Each unit of Basic Product processed further requires an additional 3 hours of warehouse labor and $5 processing cost, yielding 1 unit of Premium Basic. Each unit of Advanced Product processed further requires an additional 4 hours of warehouse labor and $6 processing cost, yielding 1 unit of Premium Advanced. Each year, LogiCorp has 8,000 hours of warehouse labor available and can purchase up to 3,000 units of raw material. How can LogiCorp maximize its profits? Assume that the cost of warehouse labor is a fixed cost, raw materials can only be purchased in whole units.
- **ground_truth**: 348500
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: LogiOR_prob_037
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_037\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_037\code.py

---

# prob_036

## question

一家小型面包房在四个城镇运营：Greenville、Springfield、Riverside 和 Hilltop。该面包房生产新鲜面包，然后将其配送到位于 Maplewood、Oakdale、Pineville、Cedarhurst 和 Brookside 的本地商店。每个面包房可供应的面包条数如表 1 所示。每家商店为满足每日顾客需求所需的面包条数如表 2 所示。面包房与商店之间的距离（以公里计）如表 3 所示。假设每条面包的配送成本（以美元计）等于起点和终点城镇之间距离的平方根，确定一种最优配送方案，使在满足所有商店需求的前提下，总运输成本最小。

表 1：各面包房可供应的面包数量  
| 面包房       | 可供应面包条数 |
|--------------|----------------|
| Greenville   | 200            |
| Springfield  | 150            |
| Riverside    | 250            |
| Hilltop      | 180            |

表 2：各商店所需的面包数量  
| 商店         | 所需面包条数   |
|--------------|----------------|
| Maplewood    | 120            |
| Oakdale      | 100            |
| Pineville    | 130            |
| Cedarhurst   | 90             |
| Brookside    | 80             |

表 3：面包房与商店之间的距离（公里）  
| 起点 / 终点  | Maplewood | Oakdale | Pineville | Cedarhurst | Brookside |
|--------------|-----------|---------|-----------|------------|-----------|
| Greenville   | 10        | 15      | 20        | 25         | 30        |
| Springfield  | 12        | 8       | 18        | 22         | 28        |
| Riverside    | 14        | 10      | 16        | 20         | 26        |
| Hilltop      | 16        | 12      | 14        | 18         | 24        |

### Other Details
- **description**: A small-scale bakery operates in four towns: Greenville, Springfield, Riverside, and Hilltop. The bakery produces fresh bread, which is then delivered to local stores located in Maplewood, Oakdale, Pineville, Cedarhurst, and Brookside. The number of loaves of bread available at each bakery is provided in Table 1. Each store requires a specific number of loaves to meet daily customer demand, as shown in Table 2. The distances (in kilometers) between the bakeries and stores are given in Table 3.  Assuming the delivery cost (in USD) for each loaf of bread is calculated as the square root of the distance between the origin and destination towns, determine the optimal delivery schedule that minimizes total transportation costs while meeting all store requirements.  

Table 1: Bread Availability at Bakeries  
| Bakery       | Loaves Available |
|--------------|------------------|
| Greenville   | 200              |
| Springfield  | 150              |
| Riverside    | 250              |
| Hilltop      | 180              |

Table 2: Bread Requirements at Stores  
| Store        | Loaves Required |
|--------------|-----------------|
| Maplewood    | 120             |
| Oakdale      | 100             |
| Pineville    | 130             |
| Cedarhurst   | 90              |
| Brookside    | 80              |

Table 3: Distances Between Bakeries and Stores (km)  
| From / To    | Maplewood | Oakdale | Pineville | Cedarhurst | Brookside |
|--------------|-----------|---------|-----------|------------|-----------|
| Greenville   | 10        | 15      | 20        | 25         | 30        |
| Springfield  | 12        | 8       | 18        | 22         | 28        |
| Riverside    | 14        | 10      | 16        | 20         | 26        |
| Hilltop      | 16        | 12      | 14        | 18         | 24        |
- **ground_truth**: 1947.670525618807
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_028
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_028\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_028\code.py

---

# prob_037

## question

一家水果加工公司 NatureBounty 专门生产两条产品线：包装鲜苹果和瓶装苹果汁。NatureBounty 使用 1（差）到 10（优）的等级对苹果进行评级。公司目前库存中有 90,000 磅 7 级苹果和 110,000 磅 4 级苹果。为保持产品质量，售卖为包装鲜果的苹果，其平均等级必须至少为 5，而用于榨汁生产的苹果，其平均等级必须至少为 6。

此外，公司还受到生产能力的限制：由于设备限制，总共最多只能加工 150,000 磅苹果。每一磅用于榨汁的苹果可产生 1.10 美元的收入，并产生 0.80 美元的变动成本；每一磅以包装形式售出的苹果可产生 0.35 美元的收入，并产生 0.12 美元的变动成本。公司还受到一项营销约束：为满足市场需求，总加工苹果中至少有 20% 必须以包装形式销售。

如何在遵守所有约束条件的前提下，帮助 NatureBounty 实现利润最大化。

### Other Details
- **description**: A fruit processing company, NatureBounty, specializes in producing two product lines: packaged fresh apples and bottled apple juice. NatureBounty grades apples on a scale of 1 (poor) to 10 (excellent). The company currently has 90,000 lbs of grade 7 apples and 110,000 lbs of grade 4 apples in inventory. To maintain product quality, the average grade of apples sold in packages must be at least 5, and the average grade of apples used for juice production must be at least 6. Additionally, the company has a production capacity constraint: it can process no more than 150,000 lbs of apples in total due to equipment limitations. Each pound of apples used for juice generates a revenue of $1.10 and incurs a variable cost of $0.80. Each pound of apples sold in packages generates a revenue of $0.35 and incurs a variable cost of $0.12. The company also has a marketing constraint: at least 20% of the total processed apples must be sold in packages to meet market demand. How to help NatureBounty maximize its profit while adhering to all constraints.
- **ground_truth**: 42900.0
- **problem_type**: LP
- **problem_size**: Toy
- **index**: LogiOR_prob_033
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_033\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_033\code.py

---

# prob_038

## question

一家物流公司需要通过选择一部分产品进行仓储，在考虑仓储空间限制和产品相容性的前提下，使利润最大化，从而优化其仓库存储。仓库总容量为 500 立方米。有 8 种可选产品，每种产品具有特定的体积、利润和相容性限制（由于安全或法规原因，某些产品不能同时存放）。产品及其属性如下所示：

| 产品编号 | 体积 (m³) | 利润 ($) | 不相容产品 |
|----------|-----------|----------|------------|
| 1        | 80        | 1200     | 3, 5       |
| 2        | 60        | 900      | 4          |
| 3        | 40        | 700      | 1, 6       |
| 4        | 70        | 1100     | 2, 7       |
| 5        | 50        | 800      | 1          |
| 6        | 30        | 500      | 3          |
| 7        | 90        | 1300     | 4          |
| 8        | 20        | 300      | 无         |

### Other Details
- **description**: A logistics company needs to optimize its warehouse storage by selecting a subset of products to store that maximizes profit while considering space constraints and product compatibility. The warehouse has a total capacity of 500 cubic meters. There are 8 products available, each with a specific volume, profit, and compatibility restrictions (some products cannot be stored together due to safety or regulatory reasons). The products and their attributes are as follows:

| Product ID | Volume (m³) | Profit ($) | Incompatible Products |
|------------|-------------|------------|------------------------|
| 1          | 80          | 1200       | 3, 5                   |
| 2          | 60          | 900        | 4                      |
| 3          | 40          | 700        | 1, 6                   |
| 4          | 70          | 1100       | 2, 7                   |
| 5          | 50          | 800        | 1                      |
| 6          | 30          | 500        | 3                      |
| 7          | 90          | 1300       | 4                      |
| 8          | 20          | 300        | None                   |
- **ground_truth**: 4200.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_057
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_057\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_057\code.py

---

# prob_039

## question

你负责管理绿科制造公司（GreenTech Manufacturing），这是一家利用再生纸生产环保包装材料的公司。公司生产两种包装产品：高端包装和标准包装。高端包装的售价为每吨 15 美元，其平均质量评分必须至少为 9；标准包装的售价为每吨 8 美元，其平均质量评分必须至少为 7。高端包装的月最大需求量为 2,500 吨，标准包装的月最大需求量为 800 吨。

再生纸可以通过三种加工方法之一进行处理。每种方法的单位加工成本和产出（收益）见下表。例如，用方法 1 处理 1 吨再生纸的成本为 4.20 美元，可产出 0.3 吨质量评分为 6 的材料、0.4 吨质量评分为 8 的材料以及 0.3 吨质量评分为 10 的材料。这些成本已包含获取再生纸的费用。在被转化为包装材料之前，质量评分为 6 和 8 的材料可以通过精制工艺进行提升：每吨支付 1.20 美元，可以将 1 吨质量评分为 6 的材料提升到质量评分 8；每吨支付 1.80 美元，可以将 1 吨质量评分为 8 的材料提升到质量评分 10。绿科制造公司应如何在满足质量和需求要求的前提下，使其月利润最大化？

| 加工方法        | 单位成本（美元/吨） | 产出（质量评分 6） | 产出（质量评分 8） | 产出（质量评分 10） |
|-----------------|----------------------|---------------------|---------------------|----------------------|
| 方法 1          | 4.20                 | 0.3                 | 0.4                 | 0.3                  |
| 方法 2          | 3.80                 | 0.5                 | 0.3                 | 0.2                  |
| 方法 3          | 4.50                 | 0.2                 | 0.5                 | 0.3                  |

### Other Details
- **description**: You are in charge of GreenTech Manufacturing, a company that produces eco-friendly packaging materials from recycled paper. The company produces two types of packaging: Premium and Standard. Premium packaging sells for $15 per ton and must have an average quality score of at least 9. Standard packaging sells for $8 per ton and must have an average quality score of at least 7. The maximum demand is 2,500 tons of Premium packaging and 800 tons of Standard packaging per month. 

Recycled paper can be processed using one of three methods. The yield and cost per ton for each method are shown in the table below. For example, processing one ton of recycled paper using Method 1 costs $4.20 and yields 0.3 tons of quality score 6, 0.4 tons of quality score 8, and 0.3 tons of quality score 10. These costs include the expenses of acquiring the recycled paper. Before being converted into packaging materials, quality scores 6 and 8 can be enhanced through a refinement process. For $1.20 per ton, one ton of quality score 6 can be upgraded to quality score 8. For $1.80 per ton, one ton of quality score 8 can be upgraded to quality score 10. How can GreenTech Manufacturing maximize its monthly profit while meeting quality and demand requirements?

| Processing Method | Cost per Ton ($) | Yield (Quality 6) | Yield (Quality 8) | Yield (Quality 10) |
|--------------------|-------------------|--------------------|--------------------|---------------------|
| Method 1           | 4.20              | 0.3                | 0.4                | 0.3                 |
| Method 2           | 3.80              | 0.5                | 0.3                | 0.2                 |
| Method 3           | 4.50              | 0.2                | 0.5                | 0.3                 |
- **ground_truth**: 28864.0
- **problem_type**: LP
- **problem_size**: Small
- **index**: LogiOR_prob_045
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_045\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_045\code.py

---

# prob_040

## question

一家区域性连锁杂货店正在推出一款新的能量饮料产品，需要对各门店的初始库存进行最优分配。共有 6 家门店（S1–S6），其特征各不相同。每家门店的已知数据包括：顾客流量（每日到店次数：S1-1200，S2-800，S3-1500，S4-900，S5-1100，S6-700）、货架空间分配（单位数：S1-50，S2-30，S3-60，S4-35，S5-45，S6-25），以及相似产品的历史销售相似度评分（0–1 量表：S1-0.8，S2-0.6，S3-0.9，S4-0.7，S5-0.75，S6-0.5）。仓库可用于初始分配的库存为 300 个单位。

有两种陈列方式可供选择：标准陈列（每家门店成本为 5 美元）和促销陈列（每家门店成本为 15 美元，但可使有效货架空间增加 20%）。总陈列预算为 120 美元。目标是最大化预期产品曝光量，其计算方式为：
（顾客流量） × （分配的单位数） × （销售相似度评分）。

### Other Details
- **description**: A regional grocery chain is launching a new energy drink product and needs to optimally allocate initial stock to its stores. There are 6 stores (S1-S6) with varying characteristics. Each store has a known customer traffic (daily visits: S1-1200, S2-800, S3-1500, S4-900, S5-1100, S6-700), shelf space allocation (units: S1-50, S2-30, S3-60, S4-35, S5-45, S6-25), and historical sales similarity score for similar products (0-1 scale: S1-0.8, S2-0.6, S3-0.9, S4-0.7, S5-0.75, S6-0.5). The warehouse has 300 units available for initial distribution. There are two types of display setups: standard (costs $5 per store) and promotional (costs $15 per store but increases effective shelf space by 20%). The total display budget is $120. The goal is to maximize expected product exposure, calculated as: (customer traffic) × (units allocated) × (sales similarity score).
- **ground_truth**: 253590.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_077
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_077\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_077\code.py

---

# prob_041

## question

一家物流公司需要为一座新建仓库安排叉车，以满足日常运营需求。未来 7 周内该仓库对叉车的需求量分别为 6、9、12、8、5、7 和 4 台。该物流公司计划从一家租赁公司租用叉车，该租赁公司提供以下几种租赁方案：

1. 长期叉车租赁：租期固定为 7 周，租金为每周 240 元。长期租赁的叉车在整个租期内均可使用。
2. 短期叉车租赁：租期灵活，可按周租赁，租金为每周 390 元。短期租赁的叉车仅在租用的当周可使用。
3. 共享叉车：这些叉车是与其他仓库联合租用的，但仅在第 2、4、5 周可用，租金为每周 220 元。
4. 促销叉车：在最后 4 周可以以优惠价格租用这些叉车，租金为每周 190 元。每周最多可租用 2 台促销叉车。

请设计一个租赁方案，在未来 7 周内完全满足仓库对叉车的需求，并使总租赁成本最小。

### Other Details
- **description**: A logistics company needs to arrange forklifts for a newly built warehouse to meet daily operational demands. The forklift requirements for the warehouse over the next 7 weeks are 6, 9, 12, 8, 5, 7 and 4. The logistics company plans to lease forklifts from a rental company, which offers the following leasing options:

1. Long-term forklift rental: The lease term is fixed at 7 weeks, with a rental cost of 240 yuan per week. Forklifts under long-term lease are available for the entire lease period.
2. Short-term forklift rental: The lease term is flexible, available on a weekly basis, with a rental cost of 390 yuan per week. Short-term leased forklifts are only available during the week they are leased.
3. Shared forklifts: These are vehicles jointly leased with other warehouses, but they are only available in weeks 2, 4, and 5. The rental cost is 220 yuan per week.
4. Promotional forklifts: These forklifts can be leased at a discounted rate during the last 4 weeks, with a rental cost of 190 yuan per week. A maximum of 2 promotional forklifts can be leased per week.

Please design a leasing plan that fully meets the warehouse's forklift requirements over the next 7 weeks while minimizing the total leasing cost.
- **ground_truth**: 14110.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_021
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_021\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_021\code.py

---

# prob_042

## question

在一个城市规划情境中，需要为公共交通站点选择位置，以服务一组由 15 个乘客需求点构成的集合，并有 5 个可供选择的站点位置。已知每个需求点到每个候选站点的距离，且仅当需求点与站点之间的距离不超过 300 米时，该需求点才被认为被此站点“覆盖”。问：至少需要选择多少个站点位置，才能保证所有需求点都被覆盖？

每个需求点与每个候选站点之间的距离矩阵如下所示：

| 需求节点\\站点 | 1    | 2    | 3    | 4    | 5    |
|-----------|------|------|------|------|------|
| 1         | 389  | 515  | 170  | 143  | 617  |
| 2         | 562  | 678  | 265  | 640  | 629  |
| 3         | 206  | 594  | 180  | 564  | 683  |
| 4         | 574  | 105  | 311  | 99   | 550  |
| 5         | 616  | 490  | 99   | 473  | 682  |
| 6         | 571  | 258  | 494  | 749  | 61   |
| 7         | 573  | 234  | 207  | 635  | 318  |
| 8         | 70   | 53   | 399  | 740  | 494  |
| 9         | 229  | 190  | 550  | 654  | 394  |
| 10        | 50   | 56   | 459  | 143  | 478  |
| 11        | 95   | 378  | 507  | 647  | 135  |
| 12        | 767  | 200  | 569  | 689  | 621  |
| 13        | 729  | 333  | 91   | 86   | 386  |
| 14        | 633  | 163  | 562  | 184  | 384  |
| 15        | 67   | 515  | 224  | 502  | 345  |

### Other Details
- **description**: In a city planning scenario, it is necessary to select locations for public transport stations to serve a set of 15 passenger demand points, with 5 potential station locations available. The distance from each demand point to each potential station is known, and a demand point is considered ""covered"" by a station only if the distance between them is no more than 300 meters. What is the minimum number of stations that need to be selected to ensure all demand points are covered?
The distance matrix between each demand point and each potential station is as follows:
| demand node\stop | 1    | 2    | 3    | 4    | 5    |
|-----------|------|------|------|------|------|
| 1         | 389  | 515  | 170  | 143  | 617  |
| 2         | 562  | 678  | 265  | 640  | 629  |
| 3         | 206  | 594  | 180  | 564  | 683  |
| 4         | 574  | 105  | 311  | 99   | 550  |
| 5         | 616  | 490  | 99   | 473  | 682  |
| 6         | 571  | 258  | 494  | 749  | 61   |
| 7         | 573  | 234  | 207  | 635  | 318  |
| 8         | 70   | 53   | 399  | 740  | 494  |
| 9         | 229  | 190  | 550  | 654  | 394  |
| 10        | 50   | 56   | 459  | 143  | 478  |
| 11        | 95   | 378  | 507  | 647  | 135  |
| 12        | 767  | 200  | 569  | 689  | 621  |
| 13        | 729  | 333  | 91   | 86   | 386  |
| 14        | 633  | 163  | 562  | 184  | 384  |
| 15        | 67   | 515  | 224  | 502  | 345  |
- **ground_truth**: 3.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_009
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_009\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_009\code.py

---

# prob_043

## question

一家制造公司运营着三个生产设施（1、2 和 3），生产两种类型的部件（X 和 Y）。每个设施在生产过程中都会产生废料，这些废料可以通过处理来减少对环境的影响。废料处理会产生成本，并减少排放的污染物数量。对设施 1 的废料进行处理的成本是每吨 12 美元，每处理 1 吨废料可以减少 0.15 吨污染物 X 和 0.35 吨污染物 Y。对设施 2 的废料进行处理的成本是每吨 8 美元，每处理 1 吨废料可以减少 0.25 吨污染物 X 和 0.20 吨污染物 Y。对设施 3 的废料进行处理的成本是每吨 18 美元，每处理 1 吨废料可以减少 0.30 吨污染物 X 和 0.40 吨污染物 Y。公司希望将污染物 X 的排放量至少减少 25 吨，并将污染物 Y 的排放量至少减少 35 吨。如何在满足污染物减排目标的前提下，使废料处理的总成本最小化？

### Other Details
- **description**: A manufacturing company operates three production facilities (1, 2, and 3) that produce two types of components (X and Y). Each facility generates waste during production, which can be treated to reduce environmental impact. Treating the waste incurs a cost and reduces the amount of pollutants emitted. It costs $12 to treat a ton of waste from facility 1, and each ton treated reduces the amount of pollutant X by 0.15 ton and the amount of pollutant Y by 0.35 ton. It costs $8 to treat a ton of waste from facility 2, and each ton treated reduces the amount of pollutant X by 0.25 ton and the amount of pollutant Y by 0.20 ton. It costs $18 to treat a ton of waste from facility 3, and each ton treated reduces the amount of pollutant X by 0.30 ton and the amount of pollutant Y by 0.40 ton. The company aims to reduce the amount of pollutant X by at least 25 tons and the amount of pollutant Y by at least 35 tons. How to minimize the total cost of treating waste while meeting the pollution reduction targets?
- **ground_truth**: 1269.5652173913045
- **problem_type**: LP
- **problem_size**: Toy
- **index**: LogiOR_prob_032
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_032\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_032\code.py

---

# prob_044

## question

# 问题描述：
一家矿业公司运营三座矿石选矿厂（P1、P2、P3），其月处理能力分别为8000、10000和12000吨。公司从四座不同的矿山（M1、M2、M3、M4）获取矿石，月可供应量分别为6000、9000、7000和5000吨。

每种矿石具有不同特性：
- 金属回收率：M1 = 85%，M2 = 92%，M3 = 78%，M4 = 88%
- 单吨处理成本：M1 = $12，M2 = $18，M3 = $10，M4 = $15
- 硫含量（杂质）：M1 = 1.2%，M2 = 0.8%，M3 = 2.1%，M4 = 1.5%
- 所有选矿厂总体平均硫含量的最高允许值：1.5%

矿山到选矿厂的单位运输成本如下表所示：
|       |  P1 |  P2 |  P3 |
|-------|-----|-----|-----|
|  M1   | $4  | $6  | $5  |
|  M2   | $7  | $5  | $8  |
|  M3   | $3  | $4  | $6  |
|  M4   | $8  | $6  | $7  |

公司需要确定从各矿山到各选矿厂的最佳矿石分配方案，在满足所有运营约束的前提下，使总金属回收量最大化，并使总成本不超过$500,000。

### Other Details
- **description**: # Problem Description:
A mining company operates three ore processing plants (P1, P2, P3) with monthly processing capacities of 8000, 10000, and 12000 tons respectively. The company sources ore from four different mines (M1, M2, M3, M4) with available quantities of 6000, 9000, 7000, and 5000 tons per month. 

Each ore type has different characteristics:
- Metal recovery rates: M1=85%, M2=92%, M3=78%, M4=88%
- Processing costs per ton: M1=$12, M2=$18, M3=$10, M4=$15
- Sulfur content (impurity): M1=1.2%, M2=0.8%, M3=2.1%, M4=1.5%
- Maximum allowed average sulfur content across all plants: 1.5%

Transportation costs per ton from mines to plants:
|       | P1  | P2  | P3  |
|-------|-----|-----|-----|
|   M1  | $4  | $6  | $5  |
|   M2  | $7  | $5  | $8  |
|   M3  | $3  | $4  | $6  |
|   M4  | $8  | $6  | $7  |

The company needs to determine the optimal ore allocation from mines to plants to maximize total metal recovery while keeping total costs under $500,000 and meeting all operational constraints.
- **ground_truth**: 22933.33333
- **problem_type**: LP
- **problem_size**: Small
- **index**: LogiOR_prob_076
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_076\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_076\code.py

---

# prob_045

## question

环球货运解决方案公司（GFS）是一家物流公司，需要利用三家承运商的服务在三个区域间运输货物。每家承运商针对大件、中件和小件货物提供不同的成本结构和服务组合，如下表所示：

| 供应商 | 每次运输成本（美元） | 大件比例（%） | 中件比例（%） | 小件比例（%） |
|--------|------------------------|---------------|----------------|---------------|
| 1      | 5.2                    | 45            | 35             | 20            |
| 2      | 4.7                    | 30            | 45             | 25            |
| 3      | 3.5                    | 15            | 20             | 65            |

GFS 必须在每个月中至少满足 500 件大件货物、300 件中件货物和 300 件小件货物的最低需求，并以总成本最小化为目标。此外，由于运力限制，每个月从任意一家供应商处承接的运输次数不得超过 700 次。求最低可能的总成本。

### Other Details
- **description**: Global Freight Solutions (GFS), a logistics company, needs to transport goods across three regions using services from three shipping providers. Each provider offers different cost structures and service mixes for large, medium, and small shipments, as detailed in Table as follows:

| Supplier | Cost Per Shipment ($) | Percent Large | Percent Medium | Percent Small |
|----------|-----------------------|---------------|----------------|---------------|
| 1        | 5.2                   | 45            | 35             | 20            |
| 2        | 4.7                   | 30            | 45             | 25            |
| 3        | 3.5                   | 15            | 20             | 65            |

GFS must fulfill a minimum monthly demand of at least 500 large shipments, 300 medium shipments, and 300 small shipments while minimizing total costs. Additionally, due to capacity constraints, no more than 700 shipments can be contracted from any single provider each month. Determine what the lowest possible cost is.
- **ground_truth**: 6553.7
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: LogiOR_prob_031
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_031\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_031\code.py

---

# prob_046

## question

CargoLink 物流公司运营两种类型的配送车辆：X 型和 Y 型。新的环境法规要求，在未来三年中投放的所有车辆，其年均碳排放量不得超过每公里 40 克。每辆 X 型车辆可产生 25,000 美元的收入，运营成本为 18,000 美元，每公里排放 70 克碳。每辆 Y 型车辆可产生 20,000 美元的收入，运营成本为 16,000 美元，每公里排放 30 克碳。由于车队运力限制，CargoLink 每年最多可投放 400 辆车辆。每种车辆在每年中可投放的最大数量如下表所示。例如，在第 3 年最多可投放 350 辆 X 型车辆。CargoLink 物流公司应如何在遵守环境法规的前提下，最大化其未来三年的总利润？

| 年度 | X 型车辆最大数量 | Y 型车辆最大数量 |
|------|--------------------|--------------------|
| 1    | 300                | 250                |
| 2    | 320                | 280                |
| 3    | 350                | 300                |

### Other Details
- **description**: CargoLink Logistics operates two types of delivery vehicles: Type X and Type Y. New environmental regulations require that the average carbon emissions of all vehicles deployed over the next three years cannot exceed 40 grams per kilometer annually. Each Type X vehicle generates $25,000 in revenue, costs $18,000 to operate, and emits 70 grams of carbon per kilometer. Each Type Y vehicle generates $20,000 in revenue, costs $16,000 to operate, and emits 30 grams of carbon per kilometer. Due to fleet capacity constraints, CargoLink can deploy a maximum of 400 vehicles each year. The maximum number of each vehicle type that can be deployed annually is shown in the table below. For example, at most, 350 Type X vehicles can be deployed in year 3. How can CargoLink Logistics maximize its total profit over the next three years while complying with environmental regulations?

| Year | Maximum Type X Vehicles | Maximum Type Y Vehicles |
|------|--------------------------|--------------------------|
| 1    | 300                      | 250                      |
| 2    | 320                      | 280                      |
| 3    | 350                      | 300                      |
- **ground_truth**: 5252000.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_044
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_044\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_044\code.py

---

# prob_047

## question

HealthSupply 公司管理着一个医疗供应链，需要在未来两个季度满足两种产品的需求：疫苗 X 和疫苗 Y。具体需求为：第 1 季度中，疫苗 X 需求为 100 剂，疫苗 Y 需求为 200 剂；第 2 季度中，疫苗 X 需求为 300 剂，疫苗 Y 需求为 100 剂。

公司每个季度最多可以生产 1,500 剂疫苗，其中每一剂疫苗 X 需要 4 升生化原材料，每一剂疫苗 Y 需要 3 升生化原材料。生化原材料的成本在第 1 季度为每升 300 美元，在第 2 季度为每升 450 美元。每个季度最多可购买 2,500 升原材料，且原材料只能在购买当季使用，不能跨季度使用。

在第 1 季度开始时，公司已有库存：疫苗 X 200 剂、疫苗 Y 300 剂。每个季度末对剩余库存按每剂 100 美元收取库存成本。

此外，公司必须满足一项监管要求：每个季度生产出来的疫苗其平均有效率必须至少达到 90%。其中，疫苗 X 的有效率为 85%，疫苗 Y 的有效率为 95%。

在此条件下，如何以最低成本（包括原材料成本和库存成本）满足各季度的需求并符合监管要求。

### Other Details
- **description**: HealthSupply Co. manages a medical supply chain that must fulfill demand for two types of products, Vaccine X and Vaccine Y, over the next two quarters, with demands of 100 doses of Vaccine X and 200 doses of Vaccine Y in Quarter 1, and 300 doses of Vaccine X and 100 doses of Vaccine Y in Quarter 2. The company can produce a maximum of 1,500 doses per quarter, with each dose of Vaccine X requiring 4 liters of raw biochemical material and each dose of Vaccine Y requiring 3 liters. The cost of raw material is $300 per liter in Quarter 1 and $450 per liter in Quarter 2, with a maximum purchase limit of 2,500 liters per quarter, and material can only be used in the quarter it is purchased. At the start of Quarter 1, the company has an inventory of 200 doses of Vaccine X and 300 doses of Vaccine Y, and a storage cost of $100 per dose is assessed at the end of each quarter for remaining inventory. Additionally, the company must meet a regulatory requirement that the vaccines produced each quarter have an average efficacy rate of at least 90%, with Vaccine X having an efficacy rate of 85% and Vaccine Y having an efficacy rate of 95%. How to meet the demand and regulatory requirements at minimum cost, including raw material and storage costs.
- **ground_truth**: 460000
- **problem_type**: MILP
- **problem_size**: Small
- **index**: LogiOR_prob_039
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_039\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_039\code.py

---

# prob_048

## question

一家全国性电子商务公司在大都市地区运营三个配送中心。每个中心目前储存的标准产品和高端产品数量如下所示：

| 配送中心           | 标准产品数量 | 高端产品数量 |
|--------------------|--------------|--------------|
| 中心 1             | 150          | 30           |
| 中心 2             | 250          | 100          |
| 中心 3             | 300          | 70           |

公司计划开设三个零售履约枢纽，为客户提供当日送达服务。公司政策要求每个履约枢纽的总库存必须严格为 300 件产品，并且每个枢纽必须储备相同数量的高端产品，以确保所有地点的客户体验一致。配送中心与履约枢纽之间的运输距离（以千米为单位）如下所示：

| 出发地 / 目的地    | 枢纽 1 | 枢纽 2 | 枢纽 3 |
|--------------------|--------|--------|--------|
| 中心 1             | 0      | 12     | 18     |
| 中心 2             | 12     | 0      | 15     |
| 中心 3             | 18     | 15     | 0      |

在满足公司库存要求的前提下，所需的最小总运输距离是多少？

### Other Details
- **description**: A national e-commerce company operates three distribution centers in the Metro region. Each center currently stores different quantities of standard and premium products as shown below:

| Distribution Center | Standard Products | Premium Products |
|---------------------|-------------------|------------------|
| Center 1            | 150               | 30               |
| Center 2            | 250               | 100              |
| Center 3            | 300               | 70               |

The company plans to open three retail fulfillment hubs that will serve customers with same-day delivery. Corporate policy requires that each fulfillment hub must maintain exactly 300 products in total inventory, and each hub must stock the same number of premium products to ensure consistent customer experience across all locations. The shipping distances (in kilometers) between distribution centers and fulfillment hubs are shown as follows:

| From / To           | Hub 1 | Hub 2 | Hub 3 |
|---------------------|-------|-------|-------|
| Center 1            | 0     | 12    | 18    |
| Center 2            | 12    | 0     | 15    |
| Center 3            | 18    | 15    | 0     |

What is the minimum total transportation distance required to move the products under the company's inventory requirements?
- **ground_truth**: 1440.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_027
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_027\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_027\code.py

---

# prob_049

## question

一家公司有三家工厂，位于不同地点，需要向位于不同地点的五家商店配送货物。第一家工厂到各商店的单位运送成本分别为 4、9、2、6 和 5，对应的储存容量为 60。第二家工厂到各商店的单位运送成本分别为 2、6、1、7 和 9，对应的储存容量为 30。第三家工厂到各商店的单位运送成本分别为 2、4、9、8 和 3，对应的储存容量为 45。五家商店的需求量分别为 15、35、20、5 和 40。应如何安排货物运输以使总运送成本最小？

### Other Details
- **description**: A company has three industries located in different locations and needs to deliver goods to five shops in different locations. The delivery costs from the first industry to the shops are 4, 9, 2, 6, and 5, respectively, with a storage capacity of 60. The delivery costs from the second industry to the shops are 2, 6, 1, 7, and 9, respectively, with a storage capacity of 30. The delivery costs from the third industry to the shops are 2, 4, 9, 8, and 3, respectively, with a storage capacity of 45. The demands of the five shops are 15, 35, 20, 5, and 40. How can the goods be transported to minimize the cost?
- **ground_truth**: 405.0
- **problem_type**: ILP
- **problem_size**: Small
- **index**: LogiOR_prob_005
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_005\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_005\code.py

---

# prob_050

## question

绿色能源解决方案公司专门生产两种可再生燃料：生物乙醇和生物柴油。公司库存中有 8,000 升原料 A 和 12,000 升原料 B。各原料的质量评分如下：原料 A——12；原料 B——6。生物乙醇的平均质量评分必须至少为 10，而生物柴油的平均质量评分必须至少为 8。

每花 1 美元用于生物乙醇的营销可产生 8 升的需求，每花 1 美元用于生物柴油的营销可产生 15 升的需求。生物乙醇的销售价格为每升 30 美元，生物柴油的销售价格为每升 25 美元。请帮助绿色能源解决方案公司使其利润最大化。假设公司不能额外购买任何一种原料。

### Other Details
- **description**: Green Energy Solutions specializes in producing two types of renewable fuel: bioethanol and biodiesel. The company has 8,000 liters of feedstock A and 12,000 liters of feedstock B in stock. The quality rating of each feedstock is as follows: feedstock A—12; feedstock B—6. Bioethanol must have an average quality rating of at least 10, while biodiesel must have an average quality rating of at least 8. The demand for each product is driven by marketing efforts. Each dollar spent on marketing bioethanol generates 8 liters of demand, and each dollar spent on marketing biodiesel generates 15 liters of demand. Bioethanol is sold for $30 per liter, and biodiesel is sold for $25 per liter. Help Green Energy Solutions maximize its profit. Assume that the company cannot purchase additional feedstock of either type.
- **ground_truth**: 518433.33333333
- **problem_type**: LP
- **problem_size**: Small
- **index**: LogiOR_prob_034
- **model_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_034\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\LogiOR\prob_034\code.py

---

# prob_051

## question

邮局正在购买盖章机，他们可以购买双型号盖章机或单型号盖章机。双型号盖章机每分钟可以盖章 50 封信件，而单型号盖章机每分钟可以盖章 30 封信件。双型号盖章机每分钟需要 20 个单位的胶水，而单型号盖章机每分钟需要 15 个单位的胶水。由于单型号盖章机更安静，单型号盖章机的数量必须多于双型号盖章机的数量。此外，邮局希望确保他们每分钟至少可以盖章 300 封信件，并且每分钟最多使用 135 个单位的胶水。他们应购买多少台每种型号的盖章机以使盖章机的总数量最小？

### Other Details
- **description**: A post office is buying stamping machines and they can buy a dual or single model stamping machine. A dual model stamping machine can stamp 50 letters per minute while a single model stamping machine can stamp 30 letters per minute. The dual model stamping machine requires 20 units of glue per minute while the single model stamping machine requires 15 units of glue per minute. Since the single model stamping machine is quieter, the number of single model stamping machines must be more than the number of dual model stamping machines. Further, the post office wants to make sure they can stamp at least 300 letters per minute and use at most 135 units of glue per minute. How many of each stamping machine should they purchase to minimize the total number of stamping machines?
- **ground_truth**: 8
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_066
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_066\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_066\code.py

---

# prob_052

## question

钙和镁存在于两种保健品中，保健品A和保健品B。一份保健品A含有30克钙和50克镁。一份保健品B含有60克钙和10克镁。保健品A每份的成本是14美元，保健品B每份的成本是25美元。某患者必须每天服用这两种保健品，以保证至少摄入400克钙和50克镁。确定该患者每天应服用多少份每种保健品，以使其每日成本最小。

### Other Details
- **description**: Calcium and Magnesium are found in two health supplements, health supplement A and health supplement B. One serving of health supplement A contains 30 grams of Calcium and 50 grams of Magnesium. One serving of health supplement B contains 60 grams of Calcium and 10 grams of Magnesium. The cost per health supplement for health supplement A is $14 and the cost per health supplement for health supplement B is $25. A patient must consume these two health supplements every day to get at least 400 grams of Calcium and 50 grams of Magnesium. Determine how much servings of each supplement the patient needs to minimize her daily cost.
- **ground_truth**: 166.66666666666669
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_197
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_197\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_197\code.py

---

# prob_053

## question

有两种工艺流程，用于给硬币镀金：工艺 A 和工艺 B。工艺 A 需要 3 单位黄金、2 根金属丝，并且可以镀 5 枚硬币。工艺 B 需要 5 单位黄金、3 根金属丝，并且可以镀 7 枚硬币。现有 500 单位黄金和 300 根金属丝可用。应分别运行多少次每种工艺流程，才能使可镀硬币的总数量最大化？

### Other Details
- **description**: There are two processes, process A and process B, to plate a coin with gold. Process A requires 3 units of gold, 2 wires, and can plate 5 coins. Process B requires 5 units of gold, 3 wires, and can plate 7 coins. There are 500 units of gold and 300 wires available. How many processes of each type should be run to maximize the total number of coins that can be plated?
- **ground_truth**: 750
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_042
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_042\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_042\code.py

---

# prob_054

## question

一家餐饮配送公司使用电动自行车或电动踏板车向顾客送餐。一辆电动自行车可以装载 8 份餐食，需要 3 单位电量；一辆电动踏板车可以装载 5 份餐食，需要 2 单位电量。由于城市对电动踏板车更加友好，电动车辆中最多只有 30% 可以是电动自行车，并且至少需要使用 20 辆电动踏板车。如果公司只有 200 单位电量可用，应当分别使用多少辆这两种车辆，才能使可配送的餐食数量最大化？

### Other Details
- **description**: A meal service company delivers meals to customers either on electric bikes or scooters. A bike can hold 8 meals and requires 3 units of charge. A scooter can hold 5 meals and requires 2 units of charge. Since the city is more friendly towards scooters, at most 30% of the electric vehicles can be bikes and at least 20 scooters must be used. If the company only has 200 units of charge available, how many of each vehicle should be used to maximize the number of meals that can be delivered?
- **ground_truth**: 513
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_154
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_154\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_154\code.py

---

# prob_055

## question

一家肉类加工厂使用两台机器——切肉机和包装机——来生产火腿和猪肋排。生产一批火腿需要在切肉机上耗费 4 小时，在包装机上耗费 2.5 小时。生产一批猪肋排需要在切肉机上耗费 2 小时，在包装机上耗费 3.5 小时。每台机器每年最多运行 4000 小时。如果每批火腿的利润为 150 美元，每批猪肋排的利润为 300 美元，应生产多少批火腿和多少批猪肋排才能使利润最大化？

### Other Details
- **description**: A meat processing plant uses two machines, a meat slicer and a meat packer, to make their hams and pork ribs. To produce one batch of hams requires 4 hours on the meat slicer and 2.5 hours on the meat packer. To produce one batch of pork ribs requires 2 hours on the meat slicer and 3.5 hours on the meat packer. Each machine runs for at most 4000 hours per year. If the profit per batch of hams is $150 and the profit per batch of pork ribs is $300, how many batches of each should be made to maximize profit?
- **ground_truth**: 342857.14285714284
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_189
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_189\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_189\code.py

---

# prob_056

## question

一场科学秀表演两种不同的实验：实验1和实验2。  
在实验1中，使用10单位的薄荷和20单位的活性成分来制作25单位的薄荷泡沫。  
在实验2中，使用12单位的薄荷和15单位的活性成分来制作18单位的薄荷泡沫。  

此外，实验1会产生5单位的黑色焦油，而实验2会产生3单位的黑色焦油。  

该表演可用的资源为：120单位的薄荷和100单位的活性成分。  
如果最多只能产生50单位的黑色焦油，应分别进行多少次实验1和实验2，才能使产生的薄荷泡沫数量最大化？

### Other Details
- **description**: A science show preforms two different demonstrations, demonstration 1 and demonstration 2. In demonstration 1, 10 units of mint and 20 units of the active ingredient is used to make 25 units of minty foam. In demonstration 2, 12 units of mint and 15 units of the active ingredient is used to make 18 units of minty foam. In addition, demonstration 1 creates 5 units of black tar while demonstration 2 creates 3 units of black tar. The show has available 120 units of mint and 100 units of active ingredients. If at most 50 units of black tar can be produced, how many of each demonstration should be done to maximize the amount of minty foam produced?
- **ground_truth**: 125
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_101
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_101\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_101\code.py

---

# prob_057

## question

一家面包店生产百吉饼和牛角面包。制作一批百吉饼需要 2 小时的烤箱时间和 0.25 小时的糕点师时间。制作一批牛角面包更为复杂，虽然只需要 1 小时的烤箱时间，但需要 2 小时的糕点师时间。一天之内，面包店最多有 70 小时可用的烤箱时间和 32 小时的糕点师时间。若使用全部可用产能，并且每批百吉饼和每批牛角面包的利润分别为 20 美元和 40 美元，面包店能够获得的最大利润是多少。

### Other Details
- **description**: A bakery bakes bagels and croissants. A batch of bagels can be made using 2 hours of oven time and 0.25 hours of pastry chef time. A batch of croissants is more complicated, so while they take 1 hour of oven time, they take 2 hours of pastry chef time. In a day, the bakery has at most 70 hours available for the oven and 32 pastry chef hours available. Using all the available capacity, what is the maximum profit the bakery can generate assuming the profit per batch is $20 and $40 respectively for a batch of bagels and a batch of croissants.
- **ground_truth**: 1072
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_036
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_036\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_036\code.py

---

# prob_058

## question

Ayse 通过混合两种肥料 C 和 Y 来生产一种植物生长化合物。该生长化合物必须至少含有 5 个单位的一氧化二氮和 8 个单位的维生素混合物。肥料 C 和 Y 的成本分别为每千克 2 美元和 3 美元。肥料 C 每千克含有 1.5 个单位的一氧化二氮和 3 个单位的维生素混合物。肥料 Y 每千克含有 5 个单位的一氧化二氮和 1 个单位的维生素混合物。确定 Ayse 的化合物的最小成本。

### Other Details
- **description**: Ayse produces a plant growth compound by mixing two types of fertilizer: C and Y. This growth compound must contain at least 5 units of nitrous oxide and 8 units of vitamin mix. Fertilizer C and Y cost $2 and $3 per kg respectively. Fertilizer C contains 1.5 units of nitrous oxide per kg and 3 units of vitamin mix per kg. Fertilizer Y contains 5 units of nitrous oxide per kg and 1 unit of vitamin mix per kg. Determine the minimum cost of Ayse's compound.
- **ground_truth**: 5.851851851851852
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_023
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_023\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_023\code.py

---

# prob_059

## question

一位在医院的病人可以服用两种不同的止痛药：止痛药1和止痛药2。每一剂止痛药1向腿部输送0.5个单位的药物，向背部输送0.8个单位的药物。每一剂止痛药2向腿部输送0.7个单位的药物，向背部输送0.4个单位的药物。此外，止痛药1还输送0.3个单位的助眠药物，止痛药2输送0.6个单位的助眠药物。助眠药物的输送量最多应为8个单位，并且向腿部输送的药物至少应为4个单位。应各服用多少剂这两种止痛药，才能使输送到背部的药物量最大化？

### Other Details
- **description**: A patient in the hospital can take two different pain killers, pain killer 1 and pain killer 2. Per dose, pain killer 1 delivers 0.5 units of medicine to the legs and 0.8 units of medicine to the back. Per dose, pain killer 2 delivers 0.7 units of medicine to the legs and 0.4 units of medicine to the back. In, addition pain killer 1 deliver 0.3 units of sleeping medicine and pain killer 2 delivers 0.6 units of sleeping medicine. At most 8 units of sleep medicine should be delivered and at least 4 units of medicine should be delivered to the legs. How many doses of each should be taken to maximize the amount of medicine delivered to the back?
- **ground_truth**: 20.8
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_119
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_119\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_119\code.py

---

# prob_060

## question

一个人服用两种补充剂来满足他每日的铁和钙需求。补充剂A的一片药含有5个单位的铁和10个单位的钙。补充剂B的一片药含有4个单位的铁和15个单位的钙。该男子每天至少需要40个单位的铁和50个单位的钙。如果补充剂A每片的价格为2美元，补充剂B每片的价格为3美元，他应分别购买多少片每种补充剂以使花费最小？

### Other Details
- **description**: A man takes two supplements to get his daily iron and calcium requirements. A pill of supplement A has 5 units of iron and 10 units of calcium. A pill of supplement B contains 4 units of iron and 15 units of calcium.  The man needs a minimum of 40 units of iron and 50 units of calcium per day. If the cost per pill of supplement A is $2 and the cost per pill of supplement B is  $3, how many of each should he buy to minimize costs?
- **ground_truth**: 16
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_201
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_201\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_201\code.py

---

# prob_061

## question

某饮用水公司将水装在玻璃瓶和塑料瓶中出售。一个玻璃瓶可以装 500 毫升的水，而一个塑料瓶可以装 750 毫升的水。由于大多数顾客更偏爱塑料瓶，塑料瓶的数量必须至少是玻璃瓶数量的 3 倍。然而，玻璃瓶的数量必须至少为 20 个。如果公司拥有 250000 毫升的水可用，为了使瓶子的总数量最大化，应生产多少个玻璃瓶和塑料瓶？

### Other Details
- **description**: A water company sells water in glass and plastic bottles. A glass bottle can hole 500 ml of water while a plastic bottle can hold 750 ml of water. Because most customer prefer plastic bottles, the number of plastic bottles must be at least 3 times the number of glass bottles. However, there must be at least 20 glass bottles. If the company has available 250000 ml of water, how many of each bottle should be made to maximize the total number of bottles?
- **ground_truth**: 363
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_077
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_077\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_077\code.py

---

# prob_062

## question

一家手工制作体育器材的制造公司生产篮球和足球。生产一个篮球需要 5 个单位的原材料和 1 小时的加工时间，而生产一个足球需要 3 个单位的原材料和 2 小时的加工时间。该制造公司共有 1500 个单位的原材料可用，工人在满负荷工作时最多可工作 750 小时。由于篮球的销售情况更好，生产的篮球数量必须至少是足球数量的三倍，但公司又希望至少生产 50 个足球。该制造公司应分别生产多少个篮球和足球，才能使生产的体育器材总数量最大？

### Other Details
- **description**: A handmade sports equipment manufacturing company makes basketballs and footballs. Basketballs require 5 units of materials and 1 hour to make whereas footballs require 3 units of materials and 2 hours to make. The manufacturing company has available 1500 units of materials and their workers working at max capacity can work for at most 750 hours. Since basketballs sell better, there must be at least three times as many basketballs as footballs but the manufacturing company would like at least 50 footballs. How many of each should the manufacturing company make to maximize the total number of sports equipment produced?
- **ground_truth**: 333
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_256
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_256\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_256\code.py

---

# prob_063

## question

一家电子产品商店销售高性能的高级台式机以及用于日常使用的普通台式机。每台高级台式机的制造成本为 2000 美元，可获得 500 美元的利润。每台普通台式机的制造成本为 1000 美元，可获得 300 美元的利润。该商店每个月最多销售 200 台台式机，并且希望在制造台式机上的支出最多为 300000 美元。应制造并销售多少台各类型的台式机才能使利润最大化？

### Other Details
- **description**: A electronics store sells premium desktops with more power as well as regular desktops for everyday use. Each premium desktop costs the store $2000 to make and yields a profit of $500. Each regular desktop costs the store $1000 to make and yields a profit of $300. The store sells at most 200 desktops each month and wants to spend at most $300000 on making the desktops. How many of each should be made and sold to maximize profit?
- **ground_truth**: 80000
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_020
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_020\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_020\code.py

---

# prob_064

## question

一家石油与天然气公司有两种类型的管道：高流量管道和低流量管道。每天，高流量管道可输送 10000 美制加仑，并建议由 12 名技术人员对该管道进行严密监控，以确保其正常运行。每天，低流量管道可输送 5000 美制加仑，并出于安全原因，应由 5 名技术人员进行严密监控。每天，该石油与天然气公司需要满足至少 150000 美制加仑的天然气需求，并且他们拥有 160 名在职技术人员。由于高流量管道具有更高的环境损害风险，高流量管道最多只能占全部管道的 35%。此外，低流量管道的数量至少为 8 根。应使用多少根每种类型的管道，才能使所需管道总数最少？

### Other Details
- **description**: An oil and gas company has two types of pipes, a high-volume and a low-volume one. Every day, the high-volume pipe allows 10000 US gallons and it is recommended that 12 technicians closely monitor the pipes to ensure that it is functioning properly. Each day, the low-volume pipe allows 5000 US gallons and 5 technicians should closely monitor for safety reasons. Every day, the oil and gas company needs to meet the demands of at least 150000 US gallons of gas and they have 160 technicians that are on their staff. Since the high-volume pipe has a higher risk of environmental damage, at most 35 percent of the pipes can be high-volume ones. Additionally, there must be a minimum of 8 low-volume pipes. How many of each pipe types should be used to reduce the total number of pipes required?
- **ground_truth**: 25
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_222
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_222\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_222\code.py

---

# prob_065

## question

一家运输公司可以购买普通货车和混合动力货车来进行配送。普通货车每天可以配送 500 个包裹，并产生 200 个单位的污染物。混合动力货车每天可以配送 300 个包裹，并产生 100 个单位的污染物。由于一项新的环保法，他们每天最多只能产生 7000 个单位的污染物。然而，公司需要每天至少能够配送 20000 个包裹。他们应购买多少辆每种类型的货车，才能使所需货车总数最少？

### Other Details
- **description**: A shipping company can purchase regular and hybrid vans to make deliveries. A regular van can deliver 500 packages per day and produces 200 units of pollutants. A hybrid van can deliver 300 packages per day and produces 100 units of pollutants. Due to a new environmental law, they can produce at most 7000 units of pollutants per day. However, the company needs to be able to deliver at least 20000 packages per day. How many of each type of van should they buy to minimize the total number of vans needed?
- **ground_truth**: 60
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_062
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_062\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_062\code.py

---

# prob_066

## question

一名外卖送餐员可以选择使用自行车或机车进行工作班次。使用自行车的一个班次可以完成 10 单订单，消耗 5 单位能量，由于速度更快，可获得 50 美元小费。使用机车的一个班次可以完成 7 单订单，消耗 6 单位能量，可获得 43 美元小费。该送餐员每月最多可工作 40 个班次，拥有 230 单位能量，并且必须至少完成 320 单订单。他至少要有 5 个机车班次，因为自行车更难获得。送餐员应在每种交通工具上安排多少个班次以使获得的小费总额最大？

### Other Details
- **description**: A food delivery person can either have shifts on bikes or scooters. A shift on a bike can deliver 10 orders, takes 5 units of energy, and brings in $50 on tips because it is faster. A shift on a scooter can deliver 7 orders, takes 6 units of energy, and brings in $43 on tips.  The delivery person has available 40 shifts a month and has 230 units of energy and must bring at least 320 orders. He must have at least 5 shifts on a scooter because bikes are harder to get. How many shifts on each type of transportation should the delivery person schedule to maximize tips received?
- **ground_truth**: 1965
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_225
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_225\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_225\code.py

---

# prob_067

## question

一家汽车制造商生产两种类型的机油：Oil Max 和 Oil Max Pro。  
一桶 Oil Max 含有 46 克物质 A、43 克物质 B 和 56 克物质 C。  
一桶 Oil Max Pro 含有 13 克物质 A、4 克物质 B 和 45 克物质 C。  

该汽车制造商拥有 1345 克物质 A、346 克物质 B、1643 克物质 C。  
此外，每桶 Oil Max 的利润为 10 美元，每桶 Oil Max Pro 的利润为 15 美元。  

该汽车制造商应生产多少桶每种机油才能使利润最大化？

### Other Details
- **description**: A car manufacturer makes two types of car oils: Oil Max and Oil Max Pro. A container of Oil Max contains 46 grams of substance A, 43 grams of substance B and 56 grams of substance C. A container of Oil Max Pro contains 13 grams of substance A, 4 grams of substance B and 45 grams of substance C. The car manufacturer has 1345 grams of substance A, 346 grams of substance B, 1643 grams of substance C. In addition, the profit per container of Oil Max is $10 and the profit per container of Oil Max Pro is $15. How many containers of each of oil should the car manufacturer make to maximize profit?
- **ground_truth**: 540
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_025
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_025\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_025\code.py

---

# prob_068

## question

一名水手在用餐时可以选择吃蟹饼或龙虾卷。他需要确保至少摄入 80 个单位的维生素 A 和 100 个单位的维生素 C。每个蟹饼含有 5 个单位的维生素 A 和 7 个单位的维生素 C。每个龙虾卷含有 8 个单位的维生素 A 和 4 个单位的维生素 C。另外，由于龙虾更昂贵，他的餐食中最多只有 40% 可以是龙虾卷。如果每个蟹饼含有 4 个单位的不饱和脂肪，而每个龙虾卷含有 6 个单位的不饱和脂肪，他应分别吃多少蟹饼和龙虾卷才能使他摄入的不饱和脂肪总量最小？

### Other Details
- **description**: A sailor can eat either a crab cakes or a lobster roll for his meals. He needs to ensure he gets at least 80 units of vitamin A and 100 units of vitamin C. Each crab cake contains 5 units of vitamin A and 7 units of vitamin C. Each lobster roll contains 8 units of vitamin A and 4 units of vitamin C. In addition, since lobster is more expensive, at most 40% of his meals should be lobster rolls. If each crab cake contains 4 units of unsaturated fat and each lobster roll contains 6 units of unsaturated fat, how many of each should he eat to minimize his unsaturated fat intake?
- **ground_truth**: 64
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_100
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_100\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_100\code.py

---

# prob_069

## question

一名男子在一片 250 英亩的湖中钓鱼，可以使用渔网或鱼线两种方式捕鱼。对湖中的每一英亩水域，使用渔网可以捕到 8 条鱼，需要 4 个单位的鱼饵，但也会给渔夫带来 2 个单位的痛苦。对湖中的每一英亩水域，使用鱼线可以捕到 5 条鱼，需要 3 个单位的鱼饵，但也会给渔夫带来 1 个单位的痛苦。渔夫共有 800 个单位的鱼饵可用，并且最多只能忍受 350 个单位的痛苦。他应当在多少英亩的水域上分别采用这两种捕鱼方式，以使他所能捕到的鱼的数量最大化？

### Other Details
- **description**: A man fishes in a 250 acre lake and can catch fish either using a net or fishing line. For each acre of the lake, using a net will catch 8 fish and requires 4 units of bait but also causes 2 units of pain for the fisherman. For each acre of the lake, using a fishing line will catch 5 fish and requires 3 units of bait but also causes 1 unit of pain for the fisherman. The fisherman has available 800 units of bait and can tolerate at most 350 units of pain. For how many acres each should he use each fishing method to maximize the amount of fish he can catch?
- **ground_truth**: 1500
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_074
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_074\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_074\code.py

---

# prob_070

## question

一位正在节食的女性需要摄入两种类型的预制餐：奶昔和蛋白棒。每份奶昔含有 2 个单位的蛋白质和 300 卡路里。每根蛋白棒含有 7 个单位的蛋白质和 250 卡路里。该女性必须摄入的蛋白棒数量是奶昔数量的 2 倍。如果该女性最多只能摄入 2000 卡路里，她应当摄入多少份奶昔和多少根蛋白棒才能使蛋白质摄入量最大化？

### Other Details
- **description**: A woman on a diet needs to eat two types of meal preps, a smoothie and a protein bar. Each smoothie contains 2 units of protein and 300 calories. Each protein bar contains 7 units of protein and 250 calories. The woman must eat 2 times more protein bars than smoothies. If the woman can consume at most 2000 calories, how many of each should she eat or drink to maximize her protein intake?
- **ground_truth**: 56
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_259
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_259\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_259\code.py

---

# prob_071

## question

一位玉米农民通过拖拉机或汽车将玉米运往城市。每辆拖拉机可以运送 40 千克玉米，而每辆汽车可以运送 20 千克玉米。由于拖拉机非常缓慢，所使用的汽车数量必须至少是所使用拖拉机数量的两倍。如果至少需要将 500 千克玉米运往城市，求在满足条件的前提下，使所需拖拉机和汽车的总数量最小。

### Other Details
- **description**: A corn farmer sends his corn to the city by either tractor or car.  A tractor can carry 40 kg of corn while a car can carry 20 kg of corn. Since tractors are very slow, the number of cars used has to be at least twice the number of tractors used. If at least 500 kg of corn need to be sent to the city, minimize the total number of tractors and cars needed.
- **ground_truth**: 19
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_166
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_166\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_166\code.py

---

# prob_072

## question

一家极其大型的滑雪度假村正在考虑购入两种类型的缆车：一种是高密度座位缆车，另一种是低密度座位缆车。高密度座位缆车每分钟可以将 45 名游客运送到山坡上，而低密度座位缆车每分钟可以运送 20 名游客。高密度座位缆车消耗 30 个单位的电力，而低密度座位缆车消耗 22 个单位的电力。由于低密度座位缆车速度较慢且对初学者更友好，因此至少要有 5 部低密度座位缆车。滑雪度假村每分钟至少需要运送 1000 名游客才能盈利，并且可用电力为 940 个单位。应计划安装多少部每种类型的缆车，才能使所需缆车的总数量最小？

### Other Details
- **description**: An extremely large ski resort is looking into purchasing two types of ski lifts, a densely-seated one and a loosely-seated one. The densely-seated ski lift is able to bring 45 guests up the slopes every minute whereas the loosely-seated ski lift can transport 20 guests every minute.  The densely-seated ski lift uses 30 units of electricity and the loosely-seated lift uses 22 units of electricity. There must be at least five loosely-seated ski lifts because they move slower and are friendlier for beginners. The ski resort needs at least 1000 guests every minute to make a profit and has available 940 units of electricity. How many of each type of ski lifts should they plan to install to minimize the total number of ski lifts needed?
- **ground_truth**: 25
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_217
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_217\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_217\code.py

---

# prob_073

## question

一位医生建议她的病人多吃鱼和鸡肉，以增加蛋白质和铁的摄入。每份鱼餐含有 10 个单位的蛋白质和 12 个单位的铁。每份鸡肉餐含有 15 个单位的蛋白质和 8 个单位的铁。该病人需要至少摄入 130 个单位的蛋白质和 120 个单位的铁。由于鸡肉餐更便宜，病人希望食用的鸡肉餐数量至少是鱼餐数量的两倍。如果每份鱼餐含有 7 个单位的脂肪，而每份鸡肉餐含有 10 个单位的脂肪，她应该各吃多少份鱼餐和鸡肉餐以使她的脂肪摄入量最小化？

### Other Details
- **description**: A doctor recommends her patient eat more fish and chicken to increase her protein and iron intake. Each fish meal contains 10 units of protein and 12 units of iron. Each chicken meal contains 15 units of protein and 8 units of iron. The patient needs to consume at least 130 units of protein and 120 units of iron. Since the chicken meal is less expensive, the patient prefers to consume at least twice as many chicken meals as fish meals. If each fish meal contains 7 units of fat and each chicken meal contains 10 units of fat, how many meals of each should she eat to minimize her fat intake?
- **ground_truth**: 118
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_096
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_096\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_096\code.py

---

# prob_074

## question

一家木工坊可以购买两种类型的电锯：斜切锯和圆锯。斜切锯每天可以切割 50 根木板，并产生 60 单位的锯末。圆锯每天可以切割 70 根木板，并产生 100 单位的锯末。木工坊每天必须至少切割 1500 根木板。然而，为了避免木工坊内污染过多，他们每天至多只能产生 2000 单位的锯末。应购买每种电锯多少台才能使所需电锯的总数量最小？

### Other Details
- **description**: A woodshop can purchase two types of saws, a miter saw and a circular saw. A miter saw can cut 50 planks of wood and produces 60 units of sawdust per day. A circular saw can cut 70 planks of wood and produces 100 units of sawdust per day. The woodshop must cut at least 1500 planks of wood per day. However, to avoid too much pollution in the woodshop they can produce at most 2000 units of sawdust per day. How many of each type of saw should be purchased to minimize the total number of saws needed?
- **ground_truth**: 26
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_043
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_043\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_043\code.py

---

# prob_075

## question

一家制药公司分批生产皮肤药膏，有普通批次和高级批次，供医院采购。生产一批普通批次需要 50 单位的药用成分和 40 单位的补水成分。生产一批高级批次需要 40 单位的药用成分和 60 单位的补水成分。公司现有 3000 单位的药用成分和 3500 单位的补水成分可供使用。由于高级批次销量更好，普通批次的数量必须少于高级批次的数量。此外，公司必须至少生产 10 批普通批次。若一批普通批次可治疗 50 人，而一批高级批次可治疗 30 人，那么应分别生产多少批普通批次和高级批次，才能使可治疗的人数最大化？

### Other Details
- **description**: A pharmaceutical company makes skin cream in batches, a regular batch and premium batch, to sell to hospitals. The regular batch requires 50 units of medicinal ingredients and 40 units of rehydration product. A premium batch requires 40 units of medicinal ingredients and 60 units of rehydration product. The company has available 3000 units of medicinal ingredients and 3500 units of rehydration product. Since the premium batch sells better, the number of regular batches must be less than the number of premium batches. In addition, the company must make at least 10 regular batches. If a regular batch can treat 50 people and a premium batch can treat 30 people, how many of each batch should be made to maximize the number of people that can be treated?
- **ground_truth**: 2650
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_097
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_097\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_097\code.py

---

# prob_076

## question

一位在北极的科学家需要通过饮用橙汁和苹果汁来在饮食中获得足够的维生素D和维生素C。每盒橙汁含有10单位的维生素D和8单位的维生素C。每盒苹果汁含有12单位的维生素D和6单位的维生素C。由于这位科学家更喜欢苹果汁，他饮用的苹果汁盒数必须至少是橙汁盒数的3倍。然而，他也必须至少饮用3盒橙汁。为了避免维生素C摄入过量，这位科学家最多只能摄入300单位的维生素C。他应分别饮用多少盒每种果汁，才能使其总维生素D摄入量最大化？

### Other Details
- **description**: A scientist in the arctic needs to get enough vitamin D and vitamin C in his diet by drinking orange and apple juice. Each box of orange juice contains 10 units of vitamin D and 8 units of vitamin C. Each box of apple juice contains 12 units of vitamin D and 6 units of vitamin C. Since the scientist prefers apple juice, he must drink at least 3 times as many apple juice boxes and orange juice boxes. However, he must also drink at least 3 orange juice boxes. To avoid a vitamin C overdose, the scientist can consume at most 300 units of vitamin C. How many of each juice box should he drink to maximize his total vitamin D intake?
- **ground_truth**: 582
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_093
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_093\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_093\code.py

---

# prob_077

## question

一家面包店制作杏仁牛角面包和开心果牛角面包。制作一只杏仁牛角面包需要 5 个单位的黄油和 8 个单位的面粉。制作一只开心果牛角面包需要 3 个单位的黄油和 6 个单位的面粉。该面包店拥有 600 个单位的黄油和 800 个单位的面粉。由于杏仁牛角面包更受欢迎，杏仁牛角面包的产量至少应为开心果牛角面包的 3 倍。如果制作一只杏仁牛角面包需要 12 分钟，而制作一只开心果牛角面包需要 10 分钟，问应制作多少只这两种牛角面包才能使总生产时间最小？

### Other Details
- **description**: A bakery makes almond and pistachio croissants. An almond croissant requires 5 units of butter and 8 units of flour. A pistachio croissant requires 3 units of butter and 6 units of flour. The bakery has available 600 units of butter and 800 units of flour. Since the almond croissant is more popular, at least 3 times as many almond croissants should be made as pistachio croissants. If making an almond croissant takes 12 minutes and making a pistachio croissant takes 10 minutes, how many of each should be made to minimize the total production time?
- **ground_truth**: 0
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_055
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_055\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_055\code.py

---

# prob_078

## question

在热带地区的一家建筑公司使用奶牛和大象来搬运砖块。一头奶牛可以在背上搬运 20 块砖，而一头大象可以在背上搬运 50 块砖。为避免大象造成过多的交通拥堵，大象的数量不能超过奶牛的数量。此外，奶牛的数量至多只能是大象数量的两倍。如果公司需要运输至少 1000 块砖，求在满足条件下可以使用的动物（奶牛和大象）的最少总数量。

### Other Details
- **description**: A construction company in the tropics uses cows and elephants to carry bricks. A cow can carry 20 bricks on its back while an elephant can carry 50 bricks on its back. To avoid having elephants create too much traffic, the number of elephant cannot exceed the number of cows. In addition, there can be at most twice the number of cows as elephants. If the company needs to transport at least 1000 bricks, find the minimum number of animals, cows and elephants, that can be used..
- **ground_truth**: 29
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_175
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_175\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_175\code.py

---

# prob_079

## question

一家热狗公司可以建造两种类型的肉铺：小型肉铺和大型肉铺。小型肉铺每天可以生产30根热狗，需要2名工人。大型肉铺每天可以生产70根热狗，需要4名工人。公司每天必须至少生产500根热狗，但他们只有30名工人可用。公司应建造多少个每种类型的肉铺，才能使肉铺的总数量最小？

### Other Details
- **description**: A hot dog company can build two types of butcher shops, a small shop and a large shop. A small shop can make 30 hot dogs per day and requires 2 workers. A large shop can make 70 hot dogs per day and requires 4 workers. The company must make at least 500 hot dogs per day but they only have available 30 workers. How many of each butcher shop should the company build to minimize the total number of butcher shops?
- **ground_truth**: 8
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_071
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_071\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_071\code.py

---

# prob_080

## question

一家清雪公司使用小型卡车和大型卡车为各个社区清除积雪。每辆小型卡车需要 2 名工人铲雪，并且可以装载 30 个单位的积雪。每辆大型卡车需要 4 名工人铲雪，并且可以装载 50 个单位的积雪。公司共有 30 名工人可用。除此之外，由于有些社区不允许大型卡车进入，因此至少有 10 辆卡车必须是小型卡车。同时，至少需要 3 辆大型卡车，并且小型卡车的数量必须是大型卡车数量的两倍。应使用多少辆每种类型的卡车，才能使可运输的积雪总量最大化？

### Other Details
- **description**: A snow removal company removes snow from neighborhoods using small trucks and large trucks. A small truck requires 2 people to shovel the snow and can carry 30 units of snow. A large truck require 4 people to shovel the snow and car carry 50 units of snow. The company has available 30 people. In addition, because some neighbourhood don’t allow big trucks, at least 10 trucks must be small. There must be at least 3 large trucks as well and the number of small trucks must be twice as much as the number of large trucks. How many of each truck should be used to maximize the total amount of snow that can be transported?
- **ground_truth**: 
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_183
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_183\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_183\code.py

---

# prob_081

## question

一家化工公司使用两种不同的方法运输氢气：高压管式拖车和液态氢罐车。第一种方法是使用高压管式拖车，每次运输可以运送 50 立方米，费用为 500 美元。第二种方法是使用液态氢罐车，每次运输可以运送 30 立方米，费用为 200 美元。公司需要运输至少 1000 立方米的氢气，并且可用预算为 3750 美元。此外，高压管式拖车运输的次数必须少于液态氢罐车运输的次数。他们应当分别使用每种运输方式多少次以使总运输次数最少？

### Other Details
- **description**: A chemical company is transporting their hydrogen using two different methods, high pressure tube trailers and liquefied hydrogen tankers. The first method is a high-pressure tube trailer which can transport 50 cubic meters each per trip at a cost of $500. The second method is using liquefied hydrogen tankers which can transport 30 cubic meters each per trip at a cost of $200. The company needs to transport at least 1000 cubic meters of hydrogen and they have budget of $3750 available. In addition, the number of transports done by the high pressure tube trailer method has to be less than the number of transports done by the liquefied hydrogen tanker method. How many of each transportation method should they use to minimize the total number of trips?
- **ground_truth**: 
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_178
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_178\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_178\code.py

---

# prob_082

## question

一位卖水商人从冰川取水，并使用小桶或大桶来运输。一个小桶可以装 40 升水，一个大桶可以装 100 升水。商人最多有 30 个小桶和 10 个大桶可用。由于小桶更容易搬运，小桶的使用数量必须至少是大桶的两倍。如果他总共最多可以运输 25 个桶，并且至少有 5 个桶必须是大桶，那么他应当使用多少个小桶和多少个大桶，才能使他能运输的冰川水总量最大？

### Other Details
- **description**: A water salesman collects water from a glacier and transports it in either small or large kegs. A small keg can hold 40 liters of water while a large keg can hold 100 liters of water. The salesman has available at most 30 small kegs and 10 large kegs. Since small kegs are easier to carry, at least twice as may small kegs must be used than large kegs. If he can transport at most 25 kegs total and at least 5 kegs must be large, how many of each should he use to maximize the total amount of glacial water he can transport?
- **ground_truth**: 1480
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_169
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_169\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_169\code.py

---

# prob_083

## question

一家糖果店将普通糖果和酸味糖果混合，制作两种产品：普通混合装和酸味惊喜混合装。每千克普通混合装包含0.8千克普通糖果和0.2千克酸味糖果。普通混合装的利润为每千克3美元。每千克酸味惊喜混合装包含0.1千克普通糖果和0.9千克酸味糖果。酸味惊喜混合装的利润为每千克5美元。该糖果店拥有80千克普通糖果和60千克酸味糖果可供使用。应当制作多少千克每种类型的糖果混合装以使利润最大化？

### Other Details
- **description**: A candy store mixes regular candy and sour candy to prepare two products, regular mix and sour surprise mix. Each kilogram of the regular mix contains 0.8 kg of regular candy and 0.2 kg of sour candy. The profit per kilogram of the regular mix is $3. Each kilogram of the sour surprise mix contains 0.1 kg of regular candy and 0.9 kg of sour candy. The profit per kilogram of the sour surprise mix is $5. The candy store has 80 kg of regular candy and 60 kg of sour candy available. How many kilograms of each type of candy mix should be created to maximize profits?
- **ground_truth**: 511.4285714285714
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_018
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_018\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_018\code.py

---

# prob_084

## question

一位零食出口商用小号和大号行李箱向客户运送零食。一个小号行李箱可以装 50 份零食，而一个大号行李箱可以装 80 份零食。大多数客户更喜欢小号行李箱，因此所使用的小号行李箱数量必须至少是大号行李箱数量的两倍。该出口商至多有 70 个小号行李箱和 50 个大号行李箱可用。如果他必须至少发送 15 个大号行李箱，并且总共最多只能发送 70 个行李箱，那么他应分别发送多少个小号和大号行李箱，才能使可运送的零食总数最大？

### Other Details
- **description**: A snack exporter sends snacks to his customer in small and large suitcases. A small suitcase can hold 50 snacks while a large suitcase can hold 80 snacks. Most customer prefer small suitcases, and so at least twice as many small suitcases must be used as large suitcases. The exporter has available at most 70 small suitcases and 50 large suitcases. If he must send at least 15 large suitcases and can send  at most 70 suitcases in total, how many of each should he send to maximize the total number of snacks that can be delivered?
- **ground_truth**: 4190
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_159
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_159\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_159\code.py

---

# prob_085

## question

从开采的矿石中提取某种金属有两种方法。第一种方法是使用工艺 J，第二种是工艺 P。工艺 J 使用 8 个单位的水可以提取 5 个单位的金属，并产生 3 个单位的污染。工艺 P 使用 6 个单位的水可以提取 9 个单位的金属，并产生 5 个单位的污染。最多可以使用 1500 个单位的水和产生 1350 个单位的污染。应各采用多少次这两种工艺，才能使提取的金属数量最大化？

### Other Details
- **description**: There are two ways to extract a metal from mined ores. The first way is to use process J and the second is process P. Process J can extract 5 units of metal using 8 units of water and produces 3 units of pollution. Process P can extract 9 units of metal using 6 units of water and produces 5 units of pollution. There can be at most 1500 units of water 1350 units of pollution. How many of each type of processes should be performed to maximize the amount of metal extracted?
- **ground_truth**: 2250
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_247
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_247\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_247\code.py

---

# prob_086

## question

一家新的快餐店制作卷饼和拼盘。每个卷饼需要 5 个单位的肉和 3 个单位的米饭。每个拼盘需要 7 个单位的肉和 5 个单位的米饭。每个卷饼需要 10 分钟制作，每个拼盘需要 8 分钟制作。该快餐店必须至少使用 3000 个单位的肉和 2500 个单位的米饭。由于卷饼更适合外带食用，因此制作的卷饼数量必须至少是拼盘数量的 3 倍。快餐店应分别制作多少个卷饼和拼盘以使总生产时间最小？

### Other Details
- **description**: A new fast food place makes wraps and platters. Each wrap requires 5 units of meat and 3 units of rice. Each platter requires 7 units of meant and 5 units of rice. While each wrap takes 10 minutes to make, each platter takes 8 minutes to make. The fast food place must use at least 3000 units of meat and 2500 units of rice. Since wraps are easier to eat on the go, at least 3 times as many wraps need to be made as platter. How many of each should the fast food place make to minimize the total production time?
- **ground_truth**: 6794
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_045
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_045\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_045\code.py

---

# prob_087

## question

一家国际货物出口商使用轮船和飞机来运输货物。每艘轮船每次可以运输相当于 40 个集装箱的货物，并在每次航程中消耗 500 升燃料。每架飞机每次可以运输相当于 20 个集装箱的货物，并在每次航程中消耗 300 升燃料。公司需要运输至少相当于 500 个集装箱的货物。此外，飞机航程次数最多为 10 次，并且至少有 50% 的航程必须由轮船承担。应安排多少次轮船航程和飞机航程才能使总燃料消耗量最小？

### Other Details
- **description**: An international goods exporter uses ships and planes to transport goods. A ship can take 40 containers worth of goods and uses 500 liters of fuel per trip. A plane can take 20 containers worth of goods and uses 300 liters of fuel per trip. The company needs to transport at least 500 containers worth of goods. In addition, there can be at most 10 plane trips made and a minimum of 50% of the trips made must be by ship. How many of each trip should be made to minimize the total amount of fuel consumed?
- **ground_truth**: 6300
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_140
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_140\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_140\code.py

---

# prob_088

## question

一家电器公司销售并安装冰箱和炉灶。每台冰箱需要 60 分钟的搬运时间和 20 分钟的安装时间。每台炉灶需要 45 分钟的搬运时间和 25 分钟的安装时间。该公司共有 20000 分钟的搬运时间和 13000 分钟的安装时间可用。如果每台冰箱的利润为 400 美元，每台炉灶的利润为 260 美元，那么他们应当销售多少台冰箱和多少台炉灶才能使利润最大化？

### Other Details
- **description**: An appliance company sells and installs refrigerators and stoves. Each refrigerator takes 60 minutes of mover time and 20 minutes of setup time. Each stove takes 45 minutes of mover time and 25 minutes of setup time. The company has available 20000 minutes of mover time and 13000 minutes of setup time. If the profit per refrigerator is $400 and the profit per stove is $260, how many of each should they sell in order to maximize profit?
- **ground_truth**: 133200
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_190
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_190\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_190\code.py

---

# prob_089

## question

一家烤芝士三明治店出售轻量烤芝士三明治和重量烤芝士三明治。  
一份轻量烤芝士三明治需要 2 片面包和 3 片芝士。  
一份重量烤芝士三明治需要 3 片面包和 5 片芝士。  

由于到店的大多数顾客都非常喜欢烤芝士三明治，店里必须制作的重量烤芝士三明治数量至少是轻量烤芝士三明治数量的 3 倍。  

店里现有 300 片面包和 500 片芝士。  

如果制作一份轻量烤芝士三明治需要 10 分钟，制作一份重量烤芝士三明治需要 15 分钟，问他们各应制作多少份这两种三明治才能使总生产时间最小？

### Other Details
- **description**: A grilled cheese shop sells a light and heavy grilled cheese sandwich. A light grilled cheese sandwich requires 2 slices of bread and 3 slices of cheese. A heavy grilled cheese sandwich requires 3 slices of bread and 5 slices of cheese. Since most people who come to the store love grilled cheese, the store must make at least 3 times as many heavy grilled cheese sandwiches as light grilled cheese sandwiches. The store has available 300 slices of bread and 500 slices of cheese. If a light grilled cheese sandwich takes 10 minutes to make and a heavy grilled cheese sandwich takes 15 minutes to make, how many of each should they make to minimize the total production time?
- **ground_truth**: 0
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_065
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_065\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_065\code.py

---

# prob_090

## question

一家汽车车身修理厂需要购买两种类型的千斤顶：一种是自动电动千斤顶，另一种是燃气驱动千斤顶。自动电动千斤顶每小时可以处理 5 辆车，并消耗 6 个单位的电力；而燃气驱动千斤顶每小时可以处理 4 辆车，并消耗 7 个单位的燃气。由于电源插座数量有限，自动电动千斤顶的数量必须少于 15 台。该修理厂最多可以使用 50 个单位的电力和 80 个单位的燃气。该修理厂应购买多少台每种类型的千斤顶，才能使每小时处理的车辆数量最大化？

### Other Details
- **description**: An autobody shop needs to purchase two types of car jacks, an automatic electric one, or a gas-powered one. The automatic electric one can process 5 cars every hour and uses 6 units of electricity whereas the gas-powered one can process 4 cars each hour using 7 units of gas. Since there is a limit to how many automatic electric ones there can be due to the limited number of power outlets, the shop must use less than 15 automatic electric ones. The shop can use at most 50 units of electricity and 80 units of gas. How many of each type of jack should the shop purchase to maximize the amount of cars processed every hour?
- **ground_truth**: 84
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_244
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_244\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_244\code.py

---

# prob_091

## question

在一个科学俱乐部里，有两张可以用来制作黏液的桌子。在桌子1上，使用3单位的粉末和5单位的胶水来制作4单位的黏液。在桌子2上，使用8单位的粉末和6单位的胶水来制作5单位的黏液。然而，桌子1会产生2单位的杂乱，而桌子2会产生4单位的杂乱。科学俱乐部现有100单位的粉末和90单位的胶水。如果最多只能产生30单位的杂乱，应分别设置多少张每种桌子，才能使所生产的黏液数量最大化？

### Other Details
- **description**: In a science club, there are two tables that can be set up to make slime. At table 1, 3 units of powder and 5 units of glue are used to make 4 units of slime. At table 2, 8 units of powder and 6 units of glue are used to make 5 units of slime. However, table 1 produces 2 units of mess while table 2 produces 4 units of mess. The science club has available 100 units of powder and 90 units of glue.  If at most 30 units of mess can be made, how many of each table should be set up to maximize the amount of slime produced?
- **ground_truth**: 60
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_121
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_121\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_121\code.py

---

# prob_092

## question

一家出租车公司计划购置车辆租给他们的司机使用。他们考虑购买摩托车或轿车。每辆摩托车可以运送 30 名乘客，产生 4 个单位的污染，并为出租车公司每班带来 100 美元的收益。每辆轿车可以运送 70 名乘客，产生 15 个单位的污染，并为公司每班带来 225 美元的收益。由于并非所有顾客都愿意乘坐摩托车，摩托车的数量最多只能占全部车辆数量的 25%。此外，公司承诺其产生的污染总量要少于 200 个单位。公司每班至少需要运送 1200 名乘客。应使用多少辆每种类型的车辆，才能使该出租车公司每班的总收益最大化？

### Other Details
- **description**: A taxi company will purchase vehicles to rent to their drivers. They are interested in purchasing either motorcycles or sedans. A motorcycle can transport 30 people, produces 4 units of pollution, and earns the taxi company $100 per shift. A sedan can transport 70 people, produces 15 units of pollution and earns the company $225 per shift. Because not every customer is comfortable with a motorcycle, at most 25% of vehicles can be motorcycles. Additionally, the company has committed to producing less than 200 units of pollution. The company needs to transport at least 1200 people every shift. How many of each type of vehicle should be used to maximize the total earnings for the taxi company per shift?
- **ground_truth**: 
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_250
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_250\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_250\code.py

---

# prob_093

## question

某机场计划购买两种用于搬运行李的车辆：四轮车和三轮车。  
一辆四轮车每天可以搬运 60 件行李，并且每天产生 30 个单位的污染物。  
一辆三轮车每天可以搬运 40 件行李，并且每天产生 15 个单位的污染物。  

机场每天需要能够搬运至少 1000 件行李。  
为避免对机场造成过度污染，每天产生的污染物最多为 430 个单位。  

问：机场应分别购买多少辆四轮车和三轮车，才能在满足上述要求的前提下，使所需车辆总数最少？

### Other Details
- **description**: An airport buys two types of vehicles, a 4-wheeler and 3-wheeler, to help move luggage. A 4-wheeler vehicle can move 60 luggage per day and produces 30 units of pollutant per day. A 3-wheeler vehicle can move 40 luggage per day and produces 15 units of pollutant per day. The airport needs to be able to move at least 1000 luggage per day. To avoid over-polluting the airport, they can produce at most 430 units of pollutant per day. How many of each vehicle should the airport buy to minimize the total number of vehicles needed.  
- **ground_truth**: 22
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_072
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_072\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_072\code.py

---

# prob_094

## question

一个村庄举办宴会，并为所有人提供自行车和汽车交通工具。一辆自行车可以载 3 人，而一辆汽车可以载 5 人。由于汽车更昂贵，汽车数量最多只能占全部车辆数量的 40%。如果该村庄需要运送至少 500 人，应使用多少辆每种交通工具，才能使所需自行车的总数量最小？

### Other Details
- **description**: A village hosts a banquet and provides bike and car transportation for everyone. A bike can take 3 people while a car can take 5 people. Since cars are more expensive, at most 40% of the vehicles can be cars. If the village needs to transport at least 500 people, how many of each vehicle should be used to minimize the total number of bikes needed?
- **ground_truth**: 80
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_167
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_167\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_167\code.py

---

# prob_095

## question

一家运输公司需要通过卡车或小汽车运送包裹。每辆卡车每次可以运送50个包裹，而每辆小汽车每次可以运送30个包裹。此外，每次卡车运输需要消耗20升汽油，而每次小汽车运输需要消耗15升汽油。卡车运输的总趟数最多为5趟，并且所有运输趟数中至少有30%必须由小汽车完成。公司需要运送至少500个包裹。应分别使用多少次卡车运输和小汽车运输，才能使汽油总消耗量最小？

### Other Details
- **description**: A shipping company need to transport packages by either truck or car. A truck can transport 50 packages per trip while a car can transport 30 packages per trip. In addition, a truck uses 20 liters of gas per trip while a car uses 15 liters of gas per trip. There can be at most 5 truck trips made and at least 30% of all the trips must be made by car. The company needs to transport at least 500 packages. How many of each transportation should they use to minimize the total amount of gas consumed?
- **ground_truth**: 230
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_180
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_180\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_180\code.py

---

# prob_096

## question

一家石油天然气公司使用集装箱和卡车将石油运送到港口。一个集装箱可以装 30 单位的石油，而一辆卡车可以装 40 单位的石油。由于政府限制，所使用的卡车数量必须至多为所使用集装箱数量的一半。如果至少需要将 2000 单位的石油运送到港口，并且至少需要使用 15 个集装箱，求在满足上述条件下，所需的集装箱和卡车总数量的最小值。

### Other Details
- **description**: An oil and gas company is sending their oil to the port using containers and trucks. A container can hold 30 units of oil while a truck can hold 40 units of oil. Due to government restrictions, the number of trucks used has to at most half the number of containers used. If at least 2000 units of oil need to be sent to the port and at least 15 containers need to be used, minimize the total number of containers and trucks needed.
- **ground_truth**: 60
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_146
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_146\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_146\code.py

---

# prob_097

## question

一家面包店制作添加纤维的布朗尼和柠檬方块。每个布朗尼需要 5 个单位的巧克力混合物和 4 个单位的纤维。每个柠檬方块需要 7 个单位的柠檬混合物和 6 个单位的纤维。柠檬方块销售速度更快，因此制作的柠檬方块数量必须大于制作的布朗尼数量。然而，为了满足所有顾客，所制作的产品中至少有 40% 必须是布朗尼。如果面包店有 2500 个单位的巧克力混合物和 3300 个单位的柠檬混合物，应各制作多少布朗尼和柠檬方块以使所需纤维的总量最小？

### Other Details
- **description**: A bakery makes fiber supplemented brownies and lemon squares. Each brownie requires 5 units of chocolate mix and 4 units of fiber. Each lemon square requires 7 units of lemon mix and 6 units of fiber. Lemon squares sell much faster and thus the number of lemon squares made must be larger than the number of brownies made. However, to please all customers, at least 40% of the items made must be brownies. If the bakery has 2500 units of chocolate mix and 3300 units of lemon mix, how many of each should be made to minimize the total amount of fiber needed?
- **ground_truth**: 26.000000000000007
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_058
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_058\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_058\code.py

---

# prob_098

## question

一位葡萄农用小箱或大箱来运输他的葡萄。一个小箱可以装 200 颗葡萄，而一个大箱可以装 500 颗。由于顾客更喜欢小箱，所使用的小箱数量必须至少是大箱数量的 3 倍。农民最多有 100 个小箱和 50 个大箱可用。此外，他的卡车最多只能装 60 个箱子，而且他必须至少使用 10 个大箱。他应当使用多少个小箱和多少个大箱，才能使所能运输的葡萄总数最大化？

### Other Details
- **description**: A grape farmer transports his grapes in either small crates or large crates. A small crate can take 200 grapes while a large crate can take 500.  Because his customers prefer smaller crates, at least 3 times as many small crates must be used than large crates. The farmer has available at most 100 small crates and at most 50 large crates. In addition, his truck can take at most 60 crates total and he must use at least 10 large crates. How many of each crate should he use to maximize the total number of grapes he can transport?
- **ground_truth**: 16500
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_179
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_179\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_179\code.py

---

# prob_099

## question

一家公司正在为其总部购买打印机，包括高端型号和普通型号。高端型号每分钟可以打印 30 页，而普通型号每分钟可以打印 20 页。此外，高端型号每分钟需要 4 个单位的墨水，而普通型号每分钟需要 3 个单位的墨水。公司希望确保每分钟至少可以打印 200 页，并且每分钟最多使用 35 个单位的墨水。由于高端型号更加方便使用，普通打印机的数量必须少于高端打印机的数量。应购买多少台这两种打印机，才能使办公室中打印机的总数量最小？

### Other Details
- **description**: An office is buying printers for their headquarters, a premium model and regular model. The premium model can print 30 pages per minute while the regular model can print 20 pages per minute. In addition, the premium model requires 4 units of ink per minute while the regular model requires 3 units of ink per minute. The office wants to make sure that at least 200 pages can be printed per minute and that at most 35 units of ink are used per minute.  Since the premium model is more user friendly, the number regular printers must be less than the number of premium printers. How many of each printer should be bought to minimize the total number of printers in the office?
- **ground_truth**: 7
- **problem_type**: ILP
- **problem_size**: Toy
- **index**: NLP4LP_prob_086
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_086\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_086\code.py

---

# prob_100

## question

一家地板公司生产强化复合地板板材和地毯。市场总监报告，每周强化复合地板板材的预期需求至少为 15,000 平方英尺，地毯的预期需求至少为 5,000 平方英尺。运输合同要求每周产品总量至少为 50,000 平方英尺。然而，由于原材料短缺，每周强化复合地板板材的产量不得超过 40,000 平方英尺，地毯的产量不得超过 20,000 平方英尺。若每平方英尺强化复合地板板材的利润为 2.1 美元，每平方英尺地毯的利润为 3.3 美元，公司每周应生产多少平方英尺的这两种产品以使利润最大化？

### Other Details
- **description**: A flooring company produces engineered laminate planks and carpets. The chief marketer reports an expected demand of at least 15,000 square feet of laminate planks and 5,000 square feet of carpets each week. The shipping contract requires a total of at least 50,000 square feet of products each week. However, due to a shortage of raw materials, no more than 40,000 square feet of laminate planks and 20,000 square feet of carpets can be produced weekly. If a square foot of laminate planks produces a $2.1 profit and a square foot of carpets yields a $3.3 profit, how many of each type of product should be made weekly to maximize the company's profit?
- **ground_truth**: 150000
- **problem_type**: LP
- **problem_size**: Toy
- **index**: NLP4LP_prob_200
- **model_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_200\model.txt
- **code_path**: data\20251129_ORThought_datasets\processed\NLP4LP\prob_200\code.py

---

