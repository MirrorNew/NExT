
\begin{algorithm}[h]
\caption{NED-Tree Construction Algorithm}
\begin{algorithmic}[1]
\REQUIRE Root Expression $expr$, Parameter Set $\mathcal{P}$, Variable Set $\mathcal{V}$
\ENSURE Definition Set $D_{\text{New}}$, Linear Form $L_{\text{FinalForm}}$

\STATE Initialize $D_{\text{New}} \leftarrow \emptyset$
\STATE $L_{\text{FinalForm}} \leftarrow \text{RecursiveBuild}(expr)$
\RETURN $D_{\text{New}}, L_{\text{FinalForm}}$

\FUNCTION{RecursiveBuild($node$)}
    \STATE \textbf{Base Case:} \IF{IsAtomic($node, \mathcal{P}, \mathcal{V}$)} \RETURN $node$ \ENDIF
    
    \STATE $children' \leftarrow$ [RecursiveBuild($c$) \textbf{for} $c$ in $node.children$]
    
    \IF{IsLinear($node.op, children'$)}
        \RETURN ConstructNode($node.op, children'$)
    \ELSE
        \STATE $node_{trans} \leftarrow \text{ST}(node.op, children')$
        \RETURN RegisterDefinition($node_{trans}, D_{\text{New}}$)
    \ENDIF
\ENDFUNCTION

\FUNCTION{RegisterDefinition($node, D$)}
    \IF{$\exists y \in D$ such that $D[y] == node$}
        \RETURN $y$
    \ENDIF
    \STATE $y_{new} \leftarrow \text{NewSymbol}("y_{temp}") $  
    \STATE $D.add(y_{new} = node)$
    \RETURN $y_{new}$
\ENDFUNCTION
\end{algorithmic}
\label{alg:ned_tree_construction}
\end{algorithm}