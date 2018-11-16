f = audioread('8677.wav');
f_cut = f(1:10000,:);
fs = 16000;
sound(f_cut,16000);
audiowrite('drum_loud.wav',5*f_cut,fs);
%%
sound(5*f_cut,16000);