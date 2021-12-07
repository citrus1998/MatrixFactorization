#!/usr/bin/perl
# https://qiita.com/ymko/items/30af1bca725cbd64d82a

my $pwd  = `pwd`; chomp $pwd;
my $file = "$pwd/script/log.txt";
my $i = `cat $file 2>/dev/null`;
$i++;
print "$i\n";

`echo $i > $file`;