a = ['prob_082', 'prob_010', 'prob_185', 'prob_081', 'prob_068', 'prob_192', 'prob_279', 'prob_106', 'prob_296', 'prob_238', 'prob_189', 'prob_255', 'prob_183', 'prob_006', 'prob_154', 'prob_197', 'prob_015', 'prob_117', 'prob_268', 'prob_239', 'prob_059', 'prob_007', 'prob_231', 'prob_285', 'prob_177', 'prob_162', 'prob_048', 'prob_245', 'prob_153', 'prob_219', 'prob_107', 'prob_137', 'prob_157', 'prob_194', 'prob_123', 'prob_149', 'prob_132', 'prob_131', 'prob_258', 'prob_290', 'prob_214', 'prob_178', 'prob_275', 'prob_142', 'prob_254', 'prob_215', 'prob_041', 'prob_150', 'prob_070', 'prob_174', 'prob_160', 'prob_097', 'prob_184', 'prob_073', 'prob_211', 'prob_274', 'prob_175', 'prob_252', 'prob_126', 'prob_035', 'prob_023', 'prob_147', 'prob_138', 'prob_179', 'prob_030', 'prob_248', 'prob_113', 'prob_291', 'prob_251', 'prob_271', 'prob_217', 'prob_155', 'prob_111', 'prob_110', 'prob_133', 'prob_014', 'prob_246', 'prob_227', 'prob_169', 'prob_144', 'prob_163', 'prob_118', 'prob_130', 'prob_136', 'prob_269', 'prob_125', 'prob_188', 'prob_262', 'prob_108', 'prob_143', 'prob_124', 'prob_204', 'prob_047', 'prob_187', 'prob_105', 'prob_139', 'prob_114', 'prob_212', 'prob_145', 'prob_019', 'prob_225', 'prob_165', 'prob_300', 'prob_151', 'prob_167', 'prob_005', 'prob_205', 'prob_199', 'prob_176', 'prob_033', 'prob_267', 'prob_115', 'prob_207', 'prob_181', 'prob_249', 'prob_109', 'prob_193', 'prob_277', 'prob_128']

def calculate_LABC_category_accuracy(results):
    """
    计算不同分类的正确率
    分类规则：
    - 001-100: L
    - 101-194: A
    - 195-244: B
    - 245-300: C

    参数:
    results: 包含分类信息的列表，每个字典包含文件名、分类和正误标志

    返回:
    一个字典，包含每个分类的正确率
    """
    # 初始化各分类的计数
    category_counts = {'L': {'err': 0, 'total': 0},
                       'A': {'err': 0, 'total': 0},
                       'B': {'err': 0, 'total': 0},
                       'C': {'err': 0, 'total': 0}}

    for item in results:
        # 获取文件名和对应的正误标志
        filename = item
        correct_flag = True

        # 提取文件名中的编号，假设格式是 "case_prob_xxx" 的文件名
        file_number = int(filename.split('_')[-1])  # 假设文件名格式类似 'case_prob_001'，提取最后一部分

        # 根据文件编号确定分类
        if 1 <= file_number <= 100:
            category = 'L'
        elif 101 <= file_number <= 194:
            category = 'A'
        elif 195 <= file_number <= 244:
            category = 'B'
        elif 245 <= file_number <= 300:
            category = 'C'
        else:
            continue  # 跳过不在1到300范围内的文件

        # 更新该分类的总文件数和正确文件数
        category_counts[category]['total'] += 1
        if correct_flag:
            category_counts[category]['err'] += 1

    # 计算每个分类的正确率
    category_accuracy = {}
    for category, counts in category_counts.items():
        if counts['total'] > 0:
            accuracy = (counts['total'] - counts['err']) / counts['total']  # 正确率 = 正确的文件数 / 总文件数
            category_accuracy[category] = accuracy
        else:
            category_accuracy[category] = 0.0  # 如果该分类没有文件，则正确率为 0

    return category_accuracy

if __name__ == '__main__':
    calculate_LABC_category_accuracy(a)