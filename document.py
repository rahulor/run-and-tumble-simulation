import os
import draw
import clean

import draw_figure1
import draw_figure2
import draw_figure4

def make_plots():
    draw.conc_profile()
    draw.initial_condition()
    draw.weights_for_grid()
    draw.runtime_grid()
    draw.score_vs_Drot()
def for_manuscript():
    draw_figure1.draw_all()
    draw_figure2.draw_all()
    draw_figure4.draw_all()
    clean.these_extensions(['.aux', '.log', '.gz', '.dvi', '.ps'])

def make_document():
    write_to_file(1)
    make_plots()
    for_manuscript()
    #
    print('Generating pdf ...')
    if os.system("pdflatex doc.tex"):
        print('Error; no pdf output')
        print('-'*21,' Close doc.pdf if it is already open!')
    else:
        print('Done!')
    #
    clean.these_extensions(['.aux', '.log', '.gz', '.dvi', '.ps'])
    if True:
        clean.remove_if_exists('figure1-eps-converted-to.pdf')
        clean.remove_if_exists('figure2-eps-converted-to.pdf')
        clean.remove_if_exists('figure4-eps-converted-to.pdf')
        clean.remove_if_exists('figures_eps/traj-eps-converted-to.pdf')
def write_to_file(nfig):
    fig_ext = '.eps'
    path = 'data\doc_macro' + '.tex'
    line1 = '\\newcommand\\nfig{'+str(nfig)+'}'
    line2 = '\\newcommand\\figext{'+ fig_ext +'}'
    f = open(path, 'w')
    f.write(line1)
    f.write('\n')
    f.write(line2)
    f.close() 

if __name__ == '__main__':
    make_document()
    