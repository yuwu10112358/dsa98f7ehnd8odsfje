function[f] = plotConfusionMatrix(pct)
num_class = size(pct, 1);
w = 400;
h = 320;
f = figure('position', [500, 250, w, h]);

gdsz = 0.25;
txt_l = 0.015;
txt_u = 0.125;
font_size = 8;
label_offset = 0.08;
ylim([-gdsz, num_class * gdsz]);
for c = 0:(num_class - 1)
    text(c * gdsz + label_offset, -gdsz + txt_u, num2str(c));
    text(-gdsz + txt_u, c * gdsz + txt_u , num2str(c));
end
text(-gdsz, 3.5 * gdsz, 'predicted class', 'rotation', 90);
text(3.5 * gdsz, -gdsz - 0.05, 'actual class');
for target_class = 0:(num_class - 1)
    for pred_class = 0:(num_class - 1)
        p = pct(target_class + 1,pred_class + 1);
        rectangle('Position', [target_class * gdsz, pred_class * gdsz, gdsz, gdsz], 'FaceColor',[1-p 1-p 1-p], 'LineStyle', 'none');
        if (p > 0.5)
            text(target_class * gdsz + txt_l, pred_class * gdsz + txt_u, num2str(100 * p, '%.1f'), 'Color', 'white', 'FontSize', font_size);
        else
            text(target_class * gdsz + txt_l, pred_class * gdsz + txt_u, num2str(100 * p, '%.1f'), 'Color', 'black', 'FontSize', font_size);
        end
    end
end
axis off;